use std::{array, iter::once, sync::Arc};

use num_traits::Zero;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        ra_poly::RaPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        mles_product_sum::eval_linear_prod_assign,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{math::Math, small_scalar::SmallScalar, thread::unsafe_allocate_zero_vec},
    zkvm::{
        bytecode::BytecodePreprocessing,
        config::OneHotParams,
        instruction::{
            CircuitFlags, Flags, InstructionFlags, InstructionLookup, InterleavedBitsMarker,
            NUM_CIRCUIT_FLAGS,
        },
        lookup_table::{LookupTables, NUM_LOOKUP_TABLES},
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::{REGISTER_COUNT, XLEN};
use itertools::{zip_eq, Itertools};
use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};
use tracer::instruction::{Cycle, Instruction};

/// Number of batched read-checking sumchecks bespokely
const N_STAGES: usize = 5;

/// Pre-computed bytecode entry flags that match Zolt's flag computation.
/// These flags differ from Jolt's `Flags` trait implementations because
/// Zolt handles instructions directly without virtual sequence expansion.
///
/// Also stores register indices as Zolt computes them (from raw instruction bits),
/// which differs from Jolt's normalized operands for B-format and S-format instructions.
#[cfg(feature = "zolt-debug")]
#[derive(Clone, Debug)]
pub struct ZoltBytecodeFlags {
    /// Circuit flags (13 bools), indexed by CircuitFlags enum
    pub circuit_flags: [bool; NUM_CIRCUIT_FLAGS],
    /// Instruction flags (7 bools), indexed by InstructionFlags enum
    pub instruction_flags: [bool; crate::zkvm::instruction::NUM_INSTRUCTION_FLAGS],
    /// Lookup table index (None = no lookup table)
    pub lookup_table_index: Option<u8>,
    /// Whether operands are interleaved (not combined arithmetically)
    pub is_interleaved: bool,
    /// Register indices as Zolt extracts them from raw instruction bits.
    /// For B-format: rd = bits[11:7] (part of immediate in standard RISC-V).
    /// For S-format: rd = bits[11:7] (part of immediate in standard RISC-V).
    /// For other formats: rd, rs1, rs2 as normally decoded.
    pub rd: Option<u8>,
    pub rs1: Option<u8>,
    pub rs2: Option<u8>,
    /// Immediate value
    pub imm: i128,
    /// ELF address
    pub address: usize,
    /// RISC-V opcode (bits[6:0]) - needed for per-opcode immediate field encoding
    pub opcode: u8,
    /// RISC-V funct3 (bits[14:12]) - needed for per-opcode immediate field encoding
    pub funct3: u8,
}

#[cfg(feature = "zolt-debug")]
impl ZoltBytecodeFlags {
    /// Encode the immediate value into a field element using the same per-opcode
    /// encoding as Zolt's stage6_prover.zig. This is critical for val polynomial
    /// consistency between prover and verifier.
    ///
    /// Three categories:
    ///   1. ADDI(funct3=0)/ADDIW(funct3=0)/JAL/JALR → unsigned u64 bitcast of sign-extended i64
    ///   2. LUI/AUIPC → truncated u32 (recovers raw instr & 0xFFFFF000)
    ///   3. Everything else → signed field value (from_i128)
    fn encode_imm_field<F: JoltField>(flags: &ZoltBytecodeFlags) -> F {
        let is_identity_add = match flags.opcode {
            0x13 => flags.funct3 == 0, // ADDI
            0x1b => flags.funct3 == 0, // ADDIW
            0x6f => true,              // JAL
            0x67 => true,              // JALR
            _ => false,
        };
        if is_identity_add {
            // Unsigned u64 bitcast of sign-extended i64 immediate
            return F::from_u64(flags.imm as i64 as u64);
        }
        let is_u_type = flags.opcode == 0x37 || flags.opcode == 0x17;
        if is_u_type {
            // LUI/AUIPC: recover the raw u32 value (instr & 0xFFFFF000)
            // flags.imm is sign-extended i32 → truncate the u64 bitcast to u32
            return F::from_u64((flags.imm as i64 as u64 as u32) as u64);
        }
        // Everything else: signed field encoding
        F::from_i128(flags.imm)
    }

    /// Compute Zolt-compatible flags from a raw 32-bit instruction word.
    /// This replicates Zolt's buildBytecodeEntries logic from stage6_prover.zig,
    /// computing flags, register indices, and lookup table from the raw instruction bits.
    pub fn from_raw_word(word: u32, address: usize) -> Self {
        use crate::zkvm::instruction::{NUM_INSTRUCTION_FLAGS, CircuitFlags as CF, InstructionFlags as IF};

        let opcode = (word & 0x7F) as u8;
        let funct3 = ((word >> 12) & 0x7) as u8;
        let funct7 = ((word >> 25) & 0x7F) as u8;
        let rd_raw = ((word >> 7) & 0x1F) as u8;
        let rs1_raw = ((word >> 15) & 0x1F) as u8;
        let rs2_raw = ((word >> 20) & 0x1F) as u8;

        let mut cf = [false; NUM_CIRCUIT_FLAGS];
        let mut inf = [false; NUM_INSTRUCTION_FLAGS];

        // Determine lookup table index (matching Zolt's getLookupTableIndex)
        let lt_idx: u8 = match opcode {
            0x33 => match funct3 { // R-type
                0 => if funct7 == 0 { 0 } // ADD → RangeCheck
                     else if funct7 == 0x20 { 0 } // SUB → RangeCheck
                     else if funct7 == 0x01 { 0 } // MUL → RangeCheck
                     else { 255 },
                7 => if funct7 == 0 { 2 } // AND → And
                     else if funct7 == 0x01 { 13 } // MULHU → UpperWord
                     else { 255 },
                6 => if funct7 == 0 { 4 } else { 255 }, // OR → Or
                4 => if funct7 == 0 { 5 } else { 255 }, // XOR → Xor
                1 => 255, // SLL - decomposed
                5 => 255, // SRL/SRA - decomposed
                2 => 10,  // SLT → SignedLessThan
                3 => 11,  // SLTU → UnsignedLessThan
                _ => 255,
            },
            0x13 => match funct3 { // I-type ALU
                0 => 0,  // ADDI → RangeCheck
                7 => 2,  // ANDI → And
                6 => 4,  // ORI → Or
                4 => 5,  // XORI → Xor
                1 => 255, // SLLI - decomposed
                5 => 255, // SRLI/SRAI - decomposed
                2 => 10,  // SLTI → SignedLessThan
                3 => 11,  // SLTIU → UnsignedLessThan
                _ => 255,
            },
            0x63 => match funct3 { // Branches
                0 => 6,  // BEQ → Equal
                1 => 9,  // BNE → NotEqual
                4 => 10, // BLT → SignedLessThan
                5 => 7,  // BGE → SignedGreaterThanEqual
                6 => 11, // BLTU → UnsignedLessThan
                7 => 8,  // BGEU → UnsignedGreaterThanEqual
                _ => 255,
            },
            0x37 => 0, // LUI → RangeCheck
            0x17 => 0, // AUIPC → RangeCheck
            0x6F => 0, // JAL → RangeCheck
            0x67 => 1, // JALR → RangeCheckAligned
            0x1b => if funct3 == 0 { 0 } else { 255 }, // ADDIW → RangeCheck
            0x3b => match funct3 { // OP-32
                0 => if funct7 == 0 { 0 }      // ADDW → RangeCheck
                     else if funct7 == 0x20 { 0 } // SUBW → RangeCheck
                     else { 255 },
                _ => 255,
            },
            _ => 255, // Load, Store, ECALL, FENCE - no lookup table
        };

        let has_lookup = lt_idx != 255;
        let lookup_table_index = if has_lookup { Some(lt_idx) } else { None };

        // Load/Store flags
        if opcode == 0x03 { cf[CF::Load as usize] = true; }
        if opcode == 0x23 { cf[CF::Store as usize] = true; }

        // Jump
        if opcode == 0x6F || opcode == 0x67 { cf[CF::Jump as usize] = true; }

        // WriteLookupOutputToRD
        // JAL (0x6F) and JALR (0x67) do NOT set this flag - they write PC+4 to rd
        // via the WritePCtoRD mechanism, not the lookup output.
        // BRANCH (0x63) also doesn't set this flag.
        if has_lookup && opcode != 0x63 && opcode != 0x6f && opcode != 0x67 {
            cf[CF::WriteLookupOutputToRD as usize] = true;
        }

        // AddOperands, SubtractOperands, MultiplyOperands
        // Must match Zolt's populateEntryFromInstruction exactly.
        // LUI, AUIPC, JAL all set AddOperands=true for identity-path lookups.
        if has_lookup {
            match opcode {
                0x33 => { // R-type
                    if funct3 == 0 && funct7 == 0 { cf[CF::AddOperands as usize] = true; }
                    if funct3 == 0 && funct7 == 0x20 { cf[CF::SubtractOperands as usize] = true; }
                    if funct7 == 0x01 && funct3 == 0 { cf[CF::MultiplyOperands as usize] = true; }
                    if funct7 == 0x01 && funct3 == 3 { cf[CF::MultiplyOperands as usize] = true; }
                },
                0x13 => { if funct3 == 0 { cf[CF::AddOperands as usize] = true; } },
                0x67 => { cf[CF::AddOperands as usize] = true; }, // JALR
                0x37 => { cf[CF::AddOperands as usize] = true; }, // LUI
                0x17 => { cf[CF::AddOperands as usize] = true; }, // AUIPC
                0x6f => { cf[CF::AddOperands as usize] = true; }, // JAL
                0x1b => { if funct3 == 0 { cf[CF::AddOperands as usize] = true; } },
                0x3b => {
                    if funct3 == 0 && funct7 == 0 { cf[CF::AddOperands as usize] = true; }
                    if funct3 == 0 && funct7 == 0x20 { cf[CF::SubtractOperands as usize] = true; }
                },
                _ => {},
            }
        }

        // Instruction flags
        // LeftOperandIsPC
        if has_lookup && (opcode == 0x17 || opcode == 0x6F) {
            inf[IF::LeftOperandIsPC as usize] = true;
        }

        // LeftOperandIsRs1Value
        if has_lookup {
            match opcode {
                0x33 | 0x13 | 0x67 | 0x63 | 0x1b | 0x3b => {
                    inf[IF::LeftOperandIsRs1Value as usize] = true;
                },
                _ => {},
            }
        }

        // RightOperandIsImm
        if has_lookup {
            match opcode {
                0x13 | 0x67 | 0x37 | 0x17 | 0x6F | 0x1b => {
                    inf[IF::RightOperandIsImm as usize] = true;
                },
                _ => {},
            }
        }

        // RightOperandIsRs2Value
        if has_lookup {
            match opcode {
                0x33 | 0x63 | 0x3b => {
                    inf[IF::RightOperandIsRs2Value as usize] = true;
                },
                _ => {},
            }
        }

        // Branch
        if opcode == 0x63 { inf[IF::Branch as usize] = true; }

        // IsRdNotZero - exclude stores (0x23) and branches (0x63)
        if rd_raw != 0 && opcode != 0x23 && opcode != 0x63 {
            inf[IF::IsRdNotZero as usize] = true;
        }

        // Compute is_interleaved
        let is_interleaved = !cf[CF::AddOperands as usize]
            && !cf[CF::SubtractOperands as usize]
            && !cf[CF::MultiplyOperands as usize]
            && !cf[CF::Advice as usize];

        // Decode immediate value (matching Zolt's DecodedInstruction.decode)
        let imm: i128 = match opcode {
            0x13 | 0x03 | 0x67 | 0x1b => { // I-type
                ((word as i32) >> 20) as i128
            },
            0x23 => { // S-type
                let imm11_5 = (word >> 25) & 0x7F;
                let imm4_0 = (word >> 7) & 0x1F;
                let raw = (imm11_5 << 5) | imm4_0;
                (((raw as i32) << 20) >> 20) as i128
            },
            0x63 => { // B-type
                let b12 = (word >> 31) & 1;
                let b11 = (word >> 7) & 1;
                let b10_5 = (word >> 25) & 0x3F;
                let b4_1 = (word >> 8) & 0xF;
                let raw = (b12 << 12) | (b11 << 11) | (b10_5 << 5) | (b4_1 << 1);
                (((raw as i32) << 19) >> 19) as i128
            },
            0x37 | 0x17 => { // U-type: imm = instruction & 0xFFFFF000 (sign-extended)
                (word & 0xFFFFF000) as i32 as i128
            },
            0x6F => { // J-type
                let b20 = (word >> 31) & 1;
                let b19_12 = (word >> 12) & 0xFF;
                let b11 = (word >> 20) & 1;
                let b10_1 = (word >> 21) & 0x3FF;
                let raw = (b20 << 20) | (b19_12 << 12) | (b11 << 11) | (b10_1 << 1);
                (((raw as i32) << 11) >> 11) as i128
            },
            0x33 | 0x3b => 0, // R-type has no immediate
            _ => 0,
        };

        // Determine rd, rs1, rs2 matching Zolt's register matrix behavior:
        // - rd:  None for S-format (0x23), B-format (0x63), and rd==0 (x0 never written)
        //   Zolt's register write matrix has `rd != 0 and rd < 32` check, so rd=0 writes
        //   are excluded. The val poly must match by setting rd=None for rd_raw==0.
        // - rs1: None for U-type (LUI 0x37, AUIPC 0x17) and J-type (JAL 0x6f)
        // - rs2: None for I-type (0x13, 0x03, 0x67, 0x1b), U-type (0x37, 0x17), J-type (0x6f)
        // When None, the val poly contribution is zero.
        let rd = match opcode {
            0x23 | 0x63 => None,
            _ if rd_raw == 0 => None,
            _ => Some(rd_raw),
        };
        let rs1 = match opcode {
            0x37 | 0x17 | 0x6f => None,
            _ => Some(rs1_raw),
        };
        let rs2 = match opcode {
            0x13 | 0x03 | 0x67 | 0x1b | 0x37 | 0x17 | 0x6f => None,
            _ => Some(rs2_raw),
        };

        ZoltBytecodeFlags {
            circuit_flags: cf,
            instruction_flags: inf,
            lookup_table_index,
            is_interleaved,
            rd,
            rs1,
            rs2,
            imm,
            address,
            opcode,
            funct3,
        }
    }

    /// Create flags for a NoOp/padding bytecode entry.
    /// Matches Jolt's Instruction::NoOp flags:
    ///   circuit_flags[DoNotUpdateUnexpandedPC] = true
    ///   instruction_flags[IsNoop] = true
    /// All other fields are zero. This is critical for BytecodeReadRaf correctness:
    /// NoOp cycles in the R1CS witness have FlagDoNotUpdateUnexpandedPC=1 and
    /// FlagIsNoop=1, so the bytecode entry must have matching flags.
    pub fn noop() -> Self {
        use crate::zkvm::instruction::{CircuitFlags, InstructionFlags, NUM_INSTRUCTION_FLAGS};
        let mut circuit_flags = [false; NUM_CIRCUIT_FLAGS];
        circuit_flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] = true;
        let mut instruction_flags = [false; NUM_INSTRUCTION_FLAGS];
        instruction_flags[InstructionFlags::IsNoop as usize] = true;
        ZoltBytecodeFlags {
            circuit_flags,
            instruction_flags,
            lookup_table_index: None,
            // NoOp has no AddOperands/SubtractOperands/MultiplyOperands/Advice flags,
            // so is_interleaved_operands() = true. This means !is_interleaved = false,
            // so noops do NOT contribute to the identity-path (InstructionRafFlag) sum.
            // This matches Stage 5's trace-based computation where NoOp cycles → not identity path.
            is_interleaved: true,
            // Use None for rd/rs1/rs2 so that noop entries contribute ZERO
            // to Stages 4 and 5 val polynomials (matching Jolt's original behavior
            // where Instruction::NoOp has operands.rd = None → zero contribution).
            // The Zolt bytecode entry uses rd=255 sentinel for the same effect.
            rd: None,
            rs1: None,
            rs2: None,
            imm: 0,
            address: 0,
            opcode: 0,
            funct3: 0,
        }
    }

    /// Create flags for bytecode entry k=0, which in Zolt's buildBytecodeEntries
    /// is overwritten successively by the three termination instructions:
    ///   1. LUI x31, upper20(termination_addr)
    ///   2. ADDI x30, x0, 1
    ///   3. SB x30, lower12(termination_addr)(x31)
    ///
    /// Critically, flags ACCUMULATE across these overwrites because Zolt only
    /// sets flags to true but never clears them between overwrites. The final
    /// entry has flags from ALL three instructions OR'd together.
    ///
    /// After this accumulation:
    ///   cf: AddOperands(from ADDI) | Store(from SB) | WriteLookupOutputToRD(from LUI,ADDI)
    ///       | VirtualInstruction | DoNotUpdateUnexpandedPC
    ///   if: RightOperandIsImm(from LUI,ADDI) | LeftOperandIsRs1Value(from ADDI)
    ///       | IsRdNotZero(from LUI: rd=31, ADDI: rd=30)
    ///   rd, rs1, rs2, imm, address: from the LAST instruction (SB)
    ///   is_interleaved: false (because AddOperands=true from ADDI)
    ///   lookup_table_index: 255 (from SB, which has no lookup)
    pub fn termination_store_k0(termination_address: u64) -> Self {
        use crate::zkvm::instruction::{NUM_INSTRUCTION_FLAGS, CircuitFlags as CF, InstructionFlags as IF};

        let upper20 = ((termination_address >> 12) & 0xFFFFF) as u32;
        let lower12 = (termination_address & 0xFFF) as u32;
        let imm_upper7 = (lower12 >> 5) & 0x7F;
        let imm_lower5 = lower12 & 0x1F;

        // Start from zero-initialized entry (matching buildBytecodeEntries initialization)
        let mut cf = [false; NUM_CIRCUIT_FLAGS];
        let mut inf = [false; NUM_INSTRUCTION_FLAGS];

        // === LUI x31 pass: opcode=0x37, has_lookup=true (RangeCheck) ===
        // WriteLookupOutputToRD (has_lookup && opcode != 0x63)
        cf[CF::WriteLookupOutputToRD as usize] = true;
        // AddOperands: LUI not in the switch → no
        // VirtualInstruction, DoNotUpdateUnexpandedPC (is_termination_store)
        cf[CF::VirtualInstruction as usize] = true;
        cf[CF::DoNotUpdateUnexpandedPC as usize] = true;
        // Instruction flags from LUI:
        // RightOperandIsImm: 0x37 in list → true
        inf[IF::RightOperandIsImm as usize] = true;
        // IsRdNotZero: rd=31, opcode=0x37 (not store/branch) → true
        inf[IF::IsRdNotZero as usize] = true;

        // === ADDI x30 pass: opcode=0x13, funct3=0, has_lookup=true (RangeCheck) ===
        // AddOperands: opcode=0x13, funct3=0 → true
        cf[CF::AddOperands as usize] = true;
        // WriteLookupOutputToRD already true
        // VirtualInstruction, DoNotUpdateUnexpandedPC already true
        // LeftOperandIsRs1Value: 0x13 in list → true
        inf[IF::LeftOperandIsRs1Value as usize] = true;
        // RightOperandIsImm already true
        // IsRdNotZero: rd=30, opcode=0x13 → true (already)

        // === SB pass: opcode=0x23, funct3=0, has_lookup=false (Store) ===
        // Store flag
        cf[CF::Store as usize] = true;
        // has_lookup=false → no new WriteLookupOutputToRD/AddOperands/etc flags set
        // But old ones remain! (This is the key: flags accumulate, never clear)
        // VirtualInstruction, DoNotUpdateUnexpandedPC already true
        // No instruction flag changes from SB (has_lookup=false, so all guarded blocks skip)
        // IsRdNotZero: rd=8, but opcode=0x23 → condition fails. Old value remains (true).

        // is_interleaved is ALWAYS set (not conditional), based on current cf state:
        let is_interleaved = !cf[CF::AddOperands as usize]
            && !cf[CF::SubtractOperands as usize]
            && !cf[CF::MultiplyOperands as usize]
            && !cf[CF::Advice as usize];

        // SB encoding for rd/rs1/rs2/imm (last instruction wins for scalar fields):
        let sb_word = (imm_upper7 << 25) | (30 << 20) | (31 << 15) | (0 << 12) | (imm_lower5 << 7) | 0x23;
        let rd_raw = ((sb_word >> 7) & 0x1F) as u8;   // imm_lower5 bits
        let rs1_raw = ((sb_word >> 15) & 0x1F) as u8;  // 31
        let rs2_raw = ((sb_word >> 20) & 0x1F) as u8;  // 30

        // SB S-type immediate
        let sb_imm11_5 = (sb_word >> 25) & 0x7F;
        let sb_imm4_0 = (sb_word >> 7) & 0x1F;
        let sb_raw = (sb_imm11_5 << 5) | sb_imm4_0;
        let imm = (((sb_raw as i32) << 20) >> 20) as i128;

        // lookup_table_index accumulates like flags: LUI sets it to RangeCheck(0),
        // ADDI keeps it at RangeCheck(0), SB has no lookup (255) so the guard
        // `if (lt_idx != 255)` keeps the old value. Final: Some(0).
        ZoltBytecodeFlags {
            circuit_flags: cf,
            instruction_flags: inf,
            lookup_table_index: Some(0),  // RangeCheck, accumulated from LUI/ADDI
            is_interleaved,
            rd: Some(rd_raw),
            rs1: Some(rs1_raw),
            rs2: Some(rs2_raw),
            imm,
            address: 0,
            opcode: 0x23, // SB (last instruction wins for scalar fields)
            funct3: 0,
        }
    }

    /// Create a termination instruction entry for non-anchor virtual instructions
    /// (LUI vsr=2, ADDI vsr=1). Sets VirtualInstruction=true and DoNotUpdateUnexpandedPC=true.
    /// Address is always 0 (unexpanded_pc=0).
    fn termination_entry_virtual(word: u32) -> Self {
        use crate::zkvm::instruction::{CircuitFlags as CF};
        let mut entry = Self::from_raw_word(word, 0);
        entry.circuit_flags[CF::VirtualInstruction as usize] = true;
        entry.circuit_flags[CF::DoNotUpdateUnexpandedPC as usize] = true;
        entry
    }

    /// Create a termination instruction entry for the anchor instruction (SB vsr=0).
    /// Only sets DoNotUpdateUnexpandedPC=true, NOT VirtualInstruction.
    /// VirtualInstruction cannot be set for the anchor because:
    ///   R1CS constraint 17: if VirtualInstruction then NextPC == PC + 1
    ///   SB is the last real cycle before NoOp padding, so NextPC=0 ≠ PC+1.
    fn termination_entry_anchor(word: u32) -> Self {
        use crate::zkvm::instruction::{CircuitFlags as CF};
        let mut entry = Self::from_raw_word(word, 0);
        entry.circuit_flags[CF::DoNotUpdateUnexpandedPC as usize] = true;
        entry
    }

    /// Create flags for a VirtualSignExtendWord instruction entry.
    /// VirtualSignExtendWord is always the last entry in a W-extension decomposition.
    /// It sign-extends the lower 32 bits of rd to 64 bits.
    /// Circuit flags: AddOperands, WriteLookupOutputToRD, VirtualInstruction
    /// Instruction flags: LeftOperandIsRs1Value, IsRdNotZero (if rd != 0)
    /// Lookup table: SignExtendHalfWord (index 21)
    /// imm = 0, rs1 = rd (reads its own output), rs2 = None
    fn virtual_sign_extend_word_entry(rd: u8, rs1: u8, address: usize, is_compressed: bool) -> Self {
        use crate::zkvm::instruction::{NUM_INSTRUCTION_FLAGS, CircuitFlags as CF, InstructionFlags as IF};
        let mut cf = [false; NUM_CIRCUIT_FLAGS];
        let mut inf = [false; NUM_INSTRUCTION_FLAGS];

        cf[CF::AddOperands as usize] = true;
        cf[CF::WriteLookupOutputToRD as usize] = true;
        cf[CF::VirtualInstruction as usize] = true;
        // VirtualSignExtendWord is always the last in the sequence (vsr=0),
        // so DoNotUpdateUnexpandedPC = false
        if is_compressed {
            cf[CF::IsCompressed as usize] = true;
        }
        // IsFirstInSequence = false (always 2nd+ entry)

        inf[IF::LeftOperandIsRs1Value as usize] = true;
        // VirtualSignExtendWord does NOT set RightOperandIsImm
        if rd != 0 {
            inf[IF::IsRdNotZero as usize] = true;
        }

        let is_interleaved = !cf[CF::AddOperands as usize]
            && !cf[CF::SubtractOperands as usize]
            && !cf[CF::MultiplyOperands as usize]
            && !cf[CF::Advice as usize];

        ZoltBytecodeFlags {
            circuit_flags: cf,
            instruction_flags: inf,
            lookup_table_index: Some(21), // SignExtendHalfWord
            is_interleaved,
            rd: if rd != 0 { Some(rd) } else { None },
            rs1: Some(rs1), // VirtualSignExtendWord reads from rs1 (may differ from rd in long sequences like REMUW)
            rs2: None,     // I-type: no rs2
            imm: 0,
            address,
            opcode: 0x0B,  // Virtual custom-0 opcode
            funct3: 0,
        }
    }

    /// Create flags for a VirtualMULI instruction entry.
    /// VirtualMULI is used as the base instruction for SLLIW decomposition.
    /// Circuit flags: MultiplyOperands, WriteLookupOutputToRD, VirtualInstruction,
    ///   DoNotUpdateUnexpandedPC (if vsr > 0), IsFirstInSequence (if first)
    /// Instruction flags: LeftOperandIsRs1Value, RightOperandIsImm, IsRdNotZero (if rd != 0)
    /// Lookup table: RangeCheck (index 0)
    fn virtual_muli_entry(rd: u8, rs1: u8, imm: i128, address: usize,
                          vsr: u16, is_first: bool, is_compressed: bool) -> Self {
        use crate::zkvm::instruction::{NUM_INSTRUCTION_FLAGS, CircuitFlags as CF, InstructionFlags as IF};
        let mut cf = [false; NUM_CIRCUIT_FLAGS];
        let mut inf = [false; NUM_INSTRUCTION_FLAGS];

        cf[CF::MultiplyOperands as usize] = true;
        cf[CF::WriteLookupOutputToRD as usize] = true;
        cf[CF::VirtualInstruction as usize] = true;
        if vsr != 0 {
            cf[CF::DoNotUpdateUnexpandedPC as usize] = true;
        }
        if is_first {
            cf[CF::IsFirstInSequence as usize] = true;
        }
        if is_compressed {
            cf[CF::IsCompressed as usize] = true;
        }

        inf[IF::LeftOperandIsRs1Value as usize] = true;
        inf[IF::RightOperandIsImm as usize] = true;
        if rd != 0 {
            inf[IF::IsRdNotZero as usize] = true;
        }

        let is_interleaved = !cf[CF::AddOperands as usize]
            && !cf[CF::SubtractOperands as usize]
            && !cf[CF::MultiplyOperands as usize]
            && !cf[CF::Advice as usize];

        ZoltBytecodeFlags {
            circuit_flags: cf,
            instruction_flags: inf,
            lookup_table_index: Some(0), // RangeCheck
            is_interleaved,
            rd: if rd != 0 { Some(rd) } else { None },
            rs1: Some(rs1),
            rs2: None,
            imm,
            address,
            opcode: 0x2B,  // Virtual custom-1 opcode
            funct3: 0,
        }
    }

    /// Create flags for a VirtualSRLI (virtual shift-right-logical) instruction entry.
    /// Circuit flags: WriteLookupOutputToRD, VirtualInstruction, (DoNotUpdateUnexpandedPC if vsr != 0)
    /// Instruction flags: LeftOperandIsRs1Value, RightOperandIsImm, IsRdNotZero (if rd != 0)
    /// Lookup table: VirtualSRL (index 26)
    /// Uses interleaved operands (NOT identity path).
    fn virtual_srli_entry(rd: u8, rs1: u8, bitmask: u64, address: usize,
                          vsr: u16, is_first: bool, is_compressed: bool) -> Self {
        use crate::zkvm::instruction::{NUM_INSTRUCTION_FLAGS, CircuitFlags as CF, InstructionFlags as IF};
        let mut cf = [false; NUM_CIRCUIT_FLAGS];
        let mut inf = [false; NUM_INSTRUCTION_FLAGS];

        cf[CF::WriteLookupOutputToRD as usize] = true;
        cf[CF::VirtualInstruction as usize] = true;
        if vsr != 0 {
            cf[CF::DoNotUpdateUnexpandedPC as usize] = true;
        }
        if is_first {
            cf[CF::IsFirstInSequence as usize] = true;
        }
        if is_compressed {
            cf[CF::IsCompressed as usize] = true;
        }

        inf[IF::LeftOperandIsRs1Value as usize] = true;
        inf[IF::RightOperandIsImm as usize] = true;
        if rd != 0 {
            inf[IF::IsRdNotZero as usize] = true;
        }

        // VirtualSRLI uses interleaved operands (no AddOperands/SubtractOperands/MultiplyOperands)
        let is_interleaved = true;

        ZoltBytecodeFlags {
            circuit_flags: cf,
            instruction_flags: inf,
            lookup_table_index: Some(26), // VirtualSRL
            is_interleaved,
            rd: if rd != 0 { Some(rd) } else { None },
            rs1: Some(rs1),
            rs2: None,
            imm: bitmask as i128,
            address,
            opcode: 0x5B,  // Virtual custom-3 opcode for SRLI
            funct3: 0,
        }
    }

    /// Create flags for a VirtualAdvice instruction entry.
    /// Circuit flags: Advice, WriteLookupOutputToRD, VirtualInstruction
    /// Instruction flags: IsRdNotZero (if rd != 0)
    /// Lookup table: RangeCheck (index 0)
    fn virtual_advice_entry(rd: u8, address: usize,
                            vsr: u16, is_first: bool, is_compressed: bool) -> Self {
        use crate::zkvm::instruction::{NUM_INSTRUCTION_FLAGS, CircuitFlags as CF, InstructionFlags as IF};
        let mut cf = [false; NUM_CIRCUIT_FLAGS];
        let mut inf = [false; NUM_INSTRUCTION_FLAGS];

        cf[CF::Advice as usize] = true;
        cf[CF::WriteLookupOutputToRD as usize] = true;
        cf[CF::VirtualInstruction as usize] = true;
        if vsr != 0 {
            cf[CF::DoNotUpdateUnexpandedPC as usize] = true;
        }
        if is_first {
            cf[CF::IsFirstInSequence as usize] = true;
        }
        if is_compressed {
            cf[CF::IsCompressed as usize] = true;
        }

        if rd != 0 {
            inf[IF::IsRdNotZero as usize] = true;
        }

        let is_interleaved = !cf[CF::AddOperands as usize]
            && !cf[CF::SubtractOperands as usize]
            && !cf[CF::MultiplyOperands as usize]
            && !cf[CF::Advice as usize];

        ZoltBytecodeFlags {
            circuit_flags: cf,
            instruction_flags: inf,
            lookup_table_index: Some(0), // RangeCheck
            is_interleaved,
            rd: if rd != 0 { Some(rd) } else { None },
            rs1: None,
            rs2: None,
            imm: 0,
            address,
            opcode: 0x02,  // VirtualAdvice opcode
            funct3: 0,
        }
    }

    /// Create flags for a VirtualAssertEQ instruction entry.
    /// Circuit flags: Assert, VirtualInstruction
    /// Instruction flags: LeftOperandIsRs1Value, RightOperandIsRs2Value
    /// Lookup table: Equal (index 6)
    fn virtual_assert_eq_entry(rs1: u8, rs2: u8, address: usize,
                               vsr: u16, is_first: bool, is_compressed: bool) -> Self {
        use crate::zkvm::instruction::{NUM_INSTRUCTION_FLAGS, CircuitFlags as CF, InstructionFlags as IF};
        let mut cf = [false; NUM_CIRCUIT_FLAGS];
        let mut inf = [false; NUM_INSTRUCTION_FLAGS];

        cf[CF::Assert as usize] = true;
        cf[CF::VirtualInstruction as usize] = true;
        if vsr != 0 {
            cf[CF::DoNotUpdateUnexpandedPC as usize] = true;
        }
        if is_first {
            cf[CF::IsFirstInSequence as usize] = true;
        }
        if is_compressed {
            cf[CF::IsCompressed as usize] = true;
        }

        inf[IF::LeftOperandIsRs1Value as usize] = true;
        inf[IF::RightOperandIsRs2Value as usize] = true;

        let is_interleaved = !cf[CF::AddOperands as usize]
            && !cf[CF::SubtractOperands as usize]
            && !cf[CF::MultiplyOperands as usize]
            && !cf[CF::Advice as usize];

        ZoltBytecodeFlags {
            circuit_flags: cf,
            instruction_flags: inf,
            lookup_table_index: Some(6), // Equal
            is_interleaved,
            rd: None,  // Assert doesn't write to rd
            rs1: Some(rs1),
            rs2: Some(rs2),
            imm: 0,
            address,
            opcode: 0x22,  // VirtualAssertEQ opcode
            funct3: 0,
        }
    }

    /// Create flags for a VirtualZeroExtendWord instruction entry.
    /// Circuit flags: AddOperands, WriteLookupOutputToRD, VirtualInstruction
    /// Instruction flags: LeftOperandIsRs1Value, IsRdNotZero (if rd != 0)
    /// Lookup table: LowerHalfWord (index 20)
    fn virtual_zero_extend_word_entry(rd: u8, rs1: u8, address: usize,
                                      vsr: u16, is_first: bool, is_compressed: bool) -> Self {
        use crate::zkvm::instruction::{NUM_INSTRUCTION_FLAGS, CircuitFlags as CF, InstructionFlags as IF};
        let mut cf = [false; NUM_CIRCUIT_FLAGS];
        let mut inf = [false; NUM_INSTRUCTION_FLAGS];

        cf[CF::AddOperands as usize] = true;
        cf[CF::WriteLookupOutputToRD as usize] = true;
        cf[CF::VirtualInstruction as usize] = true;
        if vsr != 0 {
            cf[CF::DoNotUpdateUnexpandedPC as usize] = true;
        }
        if is_first {
            cf[CF::IsFirstInSequence as usize] = true;
        }
        if is_compressed {
            cf[CF::IsCompressed as usize] = true;
        }

        inf[IF::LeftOperandIsRs1Value as usize] = true;
        if rd != 0 {
            inf[IF::IsRdNotZero as usize] = true;
        }

        let is_interleaved = !cf[CF::AddOperands as usize]
            && !cf[CF::SubtractOperands as usize]
            && !cf[CF::MultiplyOperands as usize]
            && !cf[CF::Advice as usize];

        ZoltBytecodeFlags {
            circuit_flags: cf,
            instruction_flags: inf,
            lookup_table_index: Some(20), // LowerHalfWord
            is_interleaved,
            rd: if rd != 0 { Some(rd) } else { None },
            rs1: Some(rs1),
            rs2: None,
            imm: 0,
            address,
            opcode: 0x42,  // VirtualZeroExtendWord opcode
            funct3: 0,
        }
    }

    /// Create flags for a VirtualAssertValidUnsignedRemainder instruction entry.
    /// Circuit flags: Assert, VirtualInstruction
    /// Instruction flags: LeftOperandIsRs1Value, RightOperandIsRs2Value
    /// Lookup table: ValidUnsignedRemainder (index 16)
    fn virtual_assert_valid_unsigned_remainder_entry(rs1: u8, rs2: u8, address: usize,
                                                     vsr: u16, is_first: bool, is_compressed: bool) -> Self {
        use crate::zkvm::instruction::{NUM_INSTRUCTION_FLAGS, CircuitFlags as CF, InstructionFlags as IF};
        let mut cf = [false; NUM_CIRCUIT_FLAGS];
        let mut inf = [false; NUM_INSTRUCTION_FLAGS];

        cf[CF::Assert as usize] = true;
        cf[CF::VirtualInstruction as usize] = true;
        if vsr != 0 {
            cf[CF::DoNotUpdateUnexpandedPC as usize] = true;
        }
        if is_first {
            cf[CF::IsFirstInSequence as usize] = true;
        }
        if is_compressed {
            cf[CF::IsCompressed as usize] = true;
        }

        inf[IF::LeftOperandIsRs1Value as usize] = true;
        inf[IF::RightOperandIsRs2Value as usize] = true;

        let is_interleaved = !cf[CF::AddOperands as usize]
            && !cf[CF::SubtractOperands as usize]
            && !cf[CF::MultiplyOperands as usize]
            && !cf[CF::Advice as usize];

        ZoltBytecodeFlags {
            circuit_flags: cf,
            instruction_flags: inf,
            lookup_table_index: Some(16), // ValidUnsignedRemainder
            is_interleaved,
            rd: None,  // Assert doesn't write to rd
            rs1: Some(rs1),
            rs2: Some(rs2),
            imm: 0,
            address,
            opcode: 0x62,  // VirtualAssertValidUnsignedRemainder opcode
            funct3: 0,
        }
    }

    /// Create flags for a MUL instruction within a virtual sequence.
    /// Circuit flags: MultiplyOperands, WriteLookupOutputToRD, VirtualInstruction
    /// Instruction flags: LeftOperandIsRs1Value, RightOperandIsRs2Value, IsRdNotZero (if rd != 0)
    /// Lookup table: RangeCheck (index 0)
    fn virtual_mul_entry(rd: u8, rs1: u8, rs2: u8, address: usize,
                         vsr: u16, is_first: bool, is_compressed: bool) -> Self {
        use crate::zkvm::instruction::{NUM_INSTRUCTION_FLAGS, CircuitFlags as CF, InstructionFlags as IF};
        let mut cf = [false; NUM_CIRCUIT_FLAGS];
        let mut inf = [false; NUM_INSTRUCTION_FLAGS];

        cf[CF::MultiplyOperands as usize] = true;
        cf[CF::WriteLookupOutputToRD as usize] = true;
        cf[CF::VirtualInstruction as usize] = true;
        if vsr != 0 {
            cf[CF::DoNotUpdateUnexpandedPC as usize] = true;
        }
        if is_first {
            cf[CF::IsFirstInSequence as usize] = true;
        }
        if is_compressed {
            cf[CF::IsCompressed as usize] = true;
        }

        inf[IF::LeftOperandIsRs1Value as usize] = true;
        inf[IF::RightOperandIsRs2Value as usize] = true;
        if rd != 0 {
            inf[IF::IsRdNotZero as usize] = true;
        }

        let is_interleaved = !cf[CF::AddOperands as usize]
            && !cf[CF::SubtractOperands as usize]
            && !cf[CF::MultiplyOperands as usize]
            && !cf[CF::Advice as usize];

        ZoltBytecodeFlags {
            circuit_flags: cf,
            instruction_flags: inf,
            lookup_table_index: Some(0), // RangeCheck
            is_interleaved,
            rd: if rd != 0 { Some(rd) } else { None },
            rs1: Some(rs1),
            rs2: Some(rs2),
            imm: 0,
            address,
            opcode: 0x33,  // OP (MUL opcode space)
            funct3: 0,
        }
    }

    /// Create flags for an ADD instruction within a virtual sequence.
    /// Circuit flags: AddOperands, WriteLookupOutputToRD, VirtualInstruction
    /// Instruction flags: LeftOperandIsRs1Value, RightOperandIsRs2Value, IsRdNotZero (if rd != 0)
    /// Lookup table: RangeCheck (index 0)
    fn virtual_add_entry(rd: u8, rs1: u8, rs2: u8, address: usize,
                         vsr: u16, is_first: bool, is_compressed: bool) -> Self {
        use crate::zkvm::instruction::{NUM_INSTRUCTION_FLAGS, CircuitFlags as CF, InstructionFlags as IF};
        let mut cf = [false; NUM_CIRCUIT_FLAGS];
        let mut inf = [false; NUM_INSTRUCTION_FLAGS];

        cf[CF::AddOperands as usize] = true;
        cf[CF::WriteLookupOutputToRD as usize] = true;
        cf[CF::VirtualInstruction as usize] = true;
        if vsr != 0 {
            cf[CF::DoNotUpdateUnexpandedPC as usize] = true;
        }
        if is_first {
            cf[CF::IsFirstInSequence as usize] = true;
        }
        if is_compressed {
            cf[CF::IsCompressed as usize] = true;
        }

        inf[IF::LeftOperandIsRs1Value as usize] = true;
        inf[IF::RightOperandIsRs2Value as usize] = true;
        if rd != 0 {
            inf[IF::IsRdNotZero as usize] = true;
        }

        let is_interleaved = !cf[CF::AddOperands as usize]
            && !cf[CF::SubtractOperands as usize]
            && !cf[CF::MultiplyOperands as usize]
            && !cf[CF::Advice as usize];

        ZoltBytecodeFlags {
            circuit_flags: cf,
            instruction_flags: inf,
            lookup_table_index: Some(0), // RangeCheck
            is_interleaved,
            rd: if rd != 0 { Some(rd) } else { None },
            rs1: Some(rs1),
            rs2: Some(rs2),
            imm: 0,
            address,
            opcode: 0x33,  // OP (ADD opcode space)
            funct3: 0,
        }
    }

    /// Build Zolt-compatible flags from raw ELF bytes.
    /// `raw_words` must be the same length as `bytecode`.
    /// `termination_address` is used to construct the 3 termination instruction words.
    /// Each termination instruction (LUI, ADDI, SB) gets its own bytecode entry at
    /// indices termination_base_pc, +1, +2 (matching Jolt's approach where each virtual
    /// instruction in a sequence has its own bytecode entry).
    ///
    /// W-extension instructions (ADDIW, ADDW, SUBW, MULW, SLLIW) are decomposed into
    /// virtual sequences by Jolt's inline_sequence. The bytecode array contains the
    /// expanded instructions, and raw_words contains the parent word duplicated for
    /// each expanded entry. This function detects virtual sequences using the
    /// NormalizedInstruction fields and creates appropriate entries:
    /// - Base instruction: maps opcode (0x1b→0x13, 0x3b→0x33) and sets virtual flags
    /// - VirtualSignExtendWord: custom entry with AddOperands, lookup=SignExtendHalfWord
    /// - VirtualMULI: custom entry with MultiplyOperands, lookup=RangeCheck
    pub fn from_raw_words(raw_words: &[u32], bytecode: &[Instruction], termination_address: u64) -> Vec<Self> {
        assert_eq!(raw_words.len(), bytecode.len());

        // Get termination_base_pc from global (set by preprocessing loader)
        let termination_base_pc = super::ZOLT_TERMINATION_BASE_PC.get().copied();

        // Construct the 3 termination instruction words from the termination address
        let upper20 = ((termination_address >> 12) & 0xFFFFF) as u32;
        let lower12 = (termination_address & 0xFFF) as u32;
        let imm_upper7 = (lower12 >> 5) & 0x7F;
        let imm_lower5 = lower12 & 0x1F;
        let lui_word: u32 = (upper20 << 12) | (31 << 7) | 0x37;
        let addi_word: u32 = (1 << 20) | (0 << 15) | (0 << 12) | (30 << 7) | 0x13;
        let sb_word: u32 = (imm_upper7 << 25) | (30 << 20) | (31 << 15) | (0 << 12) | (imm_lower5 << 7) | 0x23;

        use tracer::instruction::virtual_sign_extend_word::VirtualSignExtendWord;
        use tracer::instruction::virtual_muli::VirtualMULI;
        use tracer::instruction::virtual_advice::VirtualAdvice;
        use tracer::instruction::virtual_assert_eq::VirtualAssertEQ;
        use tracer::instruction::virtual_zero_extend_word::VirtualZeroExtendWord;
        use tracer::instruction::virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder;

        let mut result: Vec<Self> = raw_words.iter().zip(bytecode.iter()).enumerate().map(|(k, (word, instr))| {
            // Check if this is a termination entry
            if let Some(tbpc) = termination_base_pc {
                if k == tbpc {
                    return Self::termination_entry_virtual(lui_word);
                } else if k == tbpc + 1 {
                    return Self::termination_entry_virtual(addi_word);
                } else if k == tbpc + 2 {
                    return Self::termination_entry_anchor(sb_word);
                }
            }

            // k=0 is now just a NoOp entry (no more accumulated termination flags)
            if k == 0 || matches!(instr, Instruction::NoOp) {
                return Self::noop();
            }

            let norm = instr.normalize();

            // Check if this is a VirtualSignExtendWord instruction
            // (produced by inline_sequence for ADDIW, ADDW, SUBW, MULW, SLLIW, REMUW, etc.)
            // For 2-entry W-extension sequences (ADDIW etc.), rs1 == rd.
            // For longer sequences (REMUW), rs1 may be a virtual register (e.g., a3=33).
            if matches!(instr, Instruction::VirtualSignExtendWord(_)) {
                let rd_raw = norm.operands.rd.unwrap_or(0);
                let rs1_raw = norm.operands.rs1.unwrap_or(rd_raw);
                return Self::virtual_sign_extend_word_entry(
                    rd_raw, rs1_raw, norm.address, norm.is_compressed,
                );
            }

            // Check if this is a VirtualMULI instruction
            // (produced by inline_sequence for SLLI and SLLIW)
            if let Instruction::VirtualMULI(vmuli) = instr {
                let rd_raw = norm.operands.rd.unwrap_or(0);
                let rs1_raw = norm.operands.rs1.unwrap_or(0);
                let vsr = norm.virtual_sequence_remaining.unwrap_or(0);
                let is_first = norm.is_first_in_sequence;
                return Self::virtual_muli_entry(
                    rd_raw, rs1_raw, vmuli.operands.imm as i128, norm.address,
                    vsr, is_first, norm.is_compressed,
                );
            }

            // Check if this is a VirtualSRLI instruction
            // (produced by inline_sequence for SRLI and SRLIW)
            if let Instruction::VirtualSRLI(vsrli) = instr {
                let rd_raw = norm.operands.rd.unwrap_or(0);
                let rs1_raw = norm.operands.rs1.unwrap_or(0);
                let vsr = norm.virtual_sequence_remaining.unwrap_or(0);
                let is_first = norm.is_first_in_sequence;
                return Self::virtual_srli_entry(
                    rd_raw, rs1_raw, vsrli.operands.imm as u64, norm.address,
                    vsr, is_first, norm.is_compressed,
                );
            }

            // Check if this is a VirtualAdvice instruction
            // (produced by inline_sequence for REMUW, DIVW, REMW, etc.)
            if matches!(instr, Instruction::VirtualAdvice(_)) {
                let rd_raw = norm.operands.rd.unwrap_or(0);
                let vsr = norm.virtual_sequence_remaining.unwrap_or(0);
                let is_first = norm.is_first_in_sequence;
                return Self::virtual_advice_entry(
                    rd_raw, norm.address, vsr, is_first, norm.is_compressed,
                );
            }

            // Check if this is a VirtualAssertEQ instruction
            if matches!(instr, Instruction::VirtualAssertEQ(_)) {
                let rs1_raw = norm.operands.rs1.unwrap_or(0);
                let rs2_raw = norm.operands.rs2.unwrap_or(0);
                let vsr = norm.virtual_sequence_remaining.unwrap_or(0);
                let is_first = norm.is_first_in_sequence;
                return Self::virtual_assert_eq_entry(
                    rs1_raw, rs2_raw, norm.address, vsr, is_first, norm.is_compressed,
                );
            }

            // Check if this is a VirtualZeroExtendWord instruction
            if matches!(instr, Instruction::VirtualZeroExtendWord(_)) {
                let rd_raw = norm.operands.rd.unwrap_or(0);
                let rs1_raw = norm.operands.rs1.unwrap_or(0);
                let vsr = norm.virtual_sequence_remaining.unwrap_or(0);
                let is_first = norm.is_first_in_sequence;
                return Self::virtual_zero_extend_word_entry(
                    rd_raw, rs1_raw, norm.address, vsr, is_first, norm.is_compressed,
                );
            }

            // Check if this is a VirtualAssertValidUnsignedRemainder instruction
            if matches!(instr, Instruction::VirtualAssertValidUnsignedRemainder(_)) {
                let rs1_raw = norm.operands.rs1.unwrap_or(0);
                let rs2_raw = norm.operands.rs2.unwrap_or(0);
                let vsr = norm.virtual_sequence_remaining.unwrap_or(0);
                let is_first = norm.is_first_in_sequence;
                return Self::virtual_assert_valid_unsigned_remainder_entry(
                    rs1_raw, rs2_raw, norm.address, vsr, is_first, norm.is_compressed,
                );
            }

            // Check if this is a MUL instruction within a virtual sequence
            // (produced by inline_sequence for REMUW, DIVW, etc.)
            if matches!(instr, Instruction::MUL(_)) {
                if let Some(vsr) = norm.virtual_sequence_remaining {
                    let rd_raw = norm.operands.rd.unwrap_or(0);
                    let rs1_raw = norm.operands.rs1.unwrap_or(0);
                    let rs2_raw = norm.operands.rs2.unwrap_or(0);
                    let is_first = norm.is_first_in_sequence;
                    return Self::virtual_mul_entry(
                        rd_raw, rs1_raw, rs2_raw, norm.address, vsr, is_first, norm.is_compressed,
                    );
                }
            }

            // Check if this is an ADD instruction within a virtual sequence
            // (produced by inline_sequence for REMUW, DIVW, etc.)
            if matches!(instr, Instruction::ADD(_)) {
                if let Some(vsr) = norm.virtual_sequence_remaining {
                    let rd_raw = norm.operands.rd.unwrap_or(0);
                    let rs1_raw = norm.operands.rs1.unwrap_or(0);
                    let rs2_raw = norm.operands.rs2.unwrap_or(0);
                    let is_first = norm.is_first_in_sequence;
                    return Self::virtual_add_entry(
                        rd_raw, rs1_raw, rs2_raw, norm.address, vsr, is_first, norm.is_compressed,
                    );
                }
            }

            // For non-virtual instructions that are the first entry of a decomposed
            // W-extension sequence (e.g., ADDIW → ADDI base, ADDW → ADD base):
            // The bytecode has the base instruction (ADDI/ADD/SUB/MUL), but the
            // raw word still has the W-extension opcode (0x1b/0x3b). Map the opcode
            // and add virtual flags.
            if let Some(vsr) = norm.virtual_sequence_remaining {
                let raw_opcode = (*word & 0x7F) as u8;
                let is_w_imm32 = raw_opcode == 0x1b; // OP-IMM-32 (ADDIW, etc.)
                let is_w_op32 = raw_opcode == 0x3b;   // OP-32 (ADDW, SUBW, MULW, etc.)

                // Map opcode: 0x1b → 0x13, 0x3b → 0x33
                let mapped_word = if is_w_imm32 {
                    (*word & !0x7F) | 0x13
                } else if is_w_op32 {
                    (*word & !0x7F) | 0x33
                } else {
                    *word
                };

                let mut entry = Self::from_raw_word(mapped_word, norm.address);

                // Set virtual sequence flags
                use crate::zkvm::instruction::CircuitFlags as CF;
                entry.circuit_flags[CF::VirtualInstruction as usize] = true;
                if vsr != 0 {
                    entry.circuit_flags[CF::DoNotUpdateUnexpandedPC as usize] = true;
                }
                if norm.is_first_in_sequence {
                    entry.circuit_flags[CF::IsFirstInSequence as usize] = true;
                }
                if norm.is_compressed {
                    entry.circuit_flags[CF::IsCompressed as usize] = true;
                }
                // Recompute is_interleaved since we added flags
                entry.is_interleaved = !entry.circuit_flags[CF::AddOperands as usize]
                    && !entry.circuit_flags[CF::SubtractOperands as usize]
                    && !entry.circuit_flags[CF::MultiplyOperands as usize]
                    && !entry.circuit_flags[CF::Advice as usize];
                return entry;
            }

            // Handle standalone shift instructions that get decomposed into virtual sequences
            // at trace time. These are single-entry sequences in the bytecode, but the
            // trace execution expands them into virtual instructions. The raw word is the
            // original shift instruction, but the prover uses the EXPANDED virtual flags.
            {
                let raw_opcode = (*word & 0x7F) as u8;
                let raw_funct3 = ((*word >> 12) & 0x7) as u8;
                let rd_raw = ((*word >> 7) & 0x1F) as u8;
                let rs1_raw = ((*word >> 15) & 0x1F) as u8;

                if raw_opcode == 0x13 && raw_funct3 == 1 {
                    // SLLI → VirtualMULI (standalone single-entry virtual sequence)
                    let shamt = ((*word >> 20) & 0x3F) as u64;
                    let imm_val = 1i128 << shamt;
                    return Self::virtual_muli_entry(
                        rd_raw, rs1_raw, imm_val, norm.address,
                        0, true, norm.is_compressed,
                    );
                }

                if raw_opcode == 0x13 && raw_funct3 == 5 && ((*word >> 30) & 1) == 0 {
                    // SRLI → VirtualSRLI (standalone single-entry virtual sequence)
                    let shamt = ((*word >> 20) & 0x3F) as u64;
                    let ones: u128 = if shamt == 0 { u128::MAX } else {
                        ((1u128 << (64 - shamt)) - 1) << shamt
                    };
                    let bitmask = ones as u64;
                    return Self::virtual_srli_entry(
                        rd_raw, rs1_raw, bitmask, norm.address,
                        0, true, norm.is_compressed,
                    );
                }
            }

            // Non-virtual instruction: decode from raw word as before
            Self::from_raw_word(*word, norm.address)
        }).collect();

        // If no termination_base_pc was set, the entries beyond raw_words.len() won't
        // exist yet. They may need to be placed at indices that are within the padded
        // array. Check and fix if needed.
        if let Some(tbpc) = termination_base_pc {
            // Ensure the array is large enough
            while result.len() <= tbpc + 2 {
                result.push(Self::noop());
            }
            // Overwrite in case they weren't set in the map above
            result[tbpc] = Self::termination_entry_virtual(lui_word);
            result[tbpc + 1] = Self::termination_entry_virtual(addi_word);
            result[tbpc + 2] = Self::termination_entry_anchor(sb_word);
        }

        // Debug: dump entries in same format as Zolt
        use crate::zkvm::instruction::NUM_INSTRUCTION_FLAGS;
        eprintln!("\n[JOLT BYTECODE ENTRIES] count={} termination_base_pc={:?}", result.len(), termination_base_pc);
        let dump_count = result.len(); // dump ALL entries
        for (k, e) in result.iter().enumerate().take(dump_count) {
            let mut cf_bits: u16 = 0;
            for i in 0..NUM_CIRCUIT_FLAGS {
                if e.circuit_flags[i] { cf_bits |= 1u16 << i; }
            }
            let mut if_bits: u8 = 0;
            for i in 0..NUM_INSTRUCTION_FLAGS {
                if e.instruction_flags[i] { if_bits |= 1u8 << i; }
            }
            let lt = match e.lookup_table_index {
                Some(idx) => idx as u16,
                None => 255,
            };
            eprintln!("  entry[{:2}]: addr=0x{:08x} rd={:2} rs1={:2} rs2={:2} imm={:6} cf=0x{:04x} if=0x{:02x} lt={:3} interl={}",
                k, e.address, e.rd.unwrap_or(0), e.rs1.unwrap_or(0), e.rs2.unwrap_or(0),
                e.imm, cf_bits, if_bits, lt, e.is_interleaved as u8);
        }
        eprintln!();

        result
    }
}

/// Bytecode instruction: multi-stage Read + RAF sumcheck (N_STAGES = 5).
///
/// Stages virtualize different claim families (Stage1: Spartan outer; Stage2: product-virtualized
/// flags; Stage3: Shift; Stage4: Registers RW; Stage5: Registers val-eval + Instruction lookups).
///
/// The input claim is a γ-weighted RLC of stage rv_claims plus RAF contributions folded into
/// stages 1 and 3 via the identity polynomial. Address vars are bound in `d` chunks; cycle vars
/// are bound with per-stage `GruenSplitEqPolynomial` (low-to-high binding), producing univariates
/// of degree `d + 1` (cubic only when `d = 2`).
///
/// Challenge notation:
/// - γ: the stage-folding scalar with powers `params.gamma_powers = transcript.challenge_scalar_powers(7)`.
/// - β_s: per-stage scalars used *within* Val_s encodings (`stage{s}_gammas = transcript.challenge_scalar_powers(...)`),
///   sampled separately for each stage.
///
/// Mathematical claim:
/// - Let K = 2^{log_K} and T = 2^{log_T}.
/// - For stage s ∈ {1,2,3,4,5}, let r_s ∈ F^{log_T} and define eq_s(j) = EqPolynomial(j; r_s).
/// - Let r_addr ∈ F^{log_K}. Let ra(k, j) ∈ {0,1} be the indicator that cycle j maps to bytecode
///   row index k (i.e. `k = get_pc(cycle_j)`; this is *not* the ELF/instruction address).
///   Implemented as ∏_{i=0}^{d-1} ra_i(k_i, j) via one-hot chunking of the bytecode index k.
/// - Int(k) = 1 for all k (evaluation of the IdentityPolynomial over address variables).
/// - Define per-stage Val_s(k) (address-only) as implemented by `compute_val_*`:
///   * Stage1: Val_1(k) = unexpanded_pc(k) + β_1·imm(k) + Σ_t β_1^{2+t}·circuit_flag_t(k).
///   * Stage2: Val_2(k) = 1_{jump}(k) + β_2·1_{branch}(k) + β_2^2·rd_addr(k) + β_2^3·1_{write_lookup_to_rd}(k).
///   * Stage3: Val_3(k) = imm(k) + β_3·unexpanded_pc(k) + β_3^2·1_{L_is_rs1}(k) + β_3^3·1_{L_is_pc}(k)
///   + β_3^4·1_{R_is_rs2}(k) + β_3^5·1_{R_is_imm}(k) + β_3^6·1_{IsNoop}(k)
///   + β_3^7·1_{VirtualInstruction}(k) + β_3^8·1_{IsFirstInSequence}(k).
///   * Stage4: Val_4(k) = 1_{rd=r}(k) + β_4·1_{rs1=r}(k) + β_4^2·1_{rs2=r}(k), where r is fixed by opening.
///   * Stage5: Val_5(k) = 1_{rd=r}(k) + β_5·1_{¬interleaved}(k) + Σ_i β_5^{2+i}·1_{table=i}(k).
///
///   Here, unexpanded_pc(k) is the instruction's ELF/address field (`instr.address`) stored in the bytecode row k.
///
/// Accumulator-provided LHS (RLC of stage claims with RAF):
///   rv_1(r_1) + γ·rv_2(r_2) + γ^2·rv_3(r_3) + γ^3·rv_4(r_4) + γ^4·rv_5(r_5)
///   + γ^5·raf_1(r_1) + γ^6·raf_3(r_3).
///
/// Sumcheck RHS proved (double sum over cycles and addresses):
///   Σ_{j=0}^{T-1} Σ_{k=0}^{K-1} ra(k, j) · [
///       γ^0·eq_1(j)·Val_1(k) + γ^1·eq_2(j)·Val_2(k) + γ^2·eq_3(j)·Val_3(k)
///     + γ^3·eq_4(j)·Val_4(k) + γ^4·eq_5(j)·Val_5(k)
///     + γ^5·eq_1(j)·Int(k)   + γ^6·eq_3(j)·Int(k)
///   ].
///
/// Thus the identity established by this sumcheck is:
///   rv_1(r_1) + γ·rv_2(r_2) + γ^2·rv_3(r_3) + γ^3·rv_4(r_4) + γ^4·rv_5(r_5)
///   + γ^5·raf_1(r_1) + γ^6·raf_3(r_3)
///     = Σ_{j,k} ra(k, j) · [ Σ_{s=1}^{5} γ^{s-1}·eq_s(j)·Val_s(k) + γ^5·eq_1(j)·Int(k) + γ^6·eq_3(j)·Int(k) ].
///
/// Binding/implementation notes:
/// - Address variables are bound first (low→high in the sumcheck binding order) in `d` chunks,
///   accumulating `F_i` and `v` tables;
///   this materializes the address-only Val_s(k) evaluations and sets up `ra_i` polynomials.
/// - Cycle variables are then bound (low→high) per stage with `GruenSplitEqPolynomial`, using
///   previous-round claims to recover the degree-(d+1) univariate each round.
/// - RAF injection uses `VirtualPolynomial::PC` (not `UnexpandedPC`): `raf_claim` comes from
///   `SumcheckId::SpartanOuter` and `raf_shift_claim` from `SumcheckId::SpartanShift`.
/// - The Stage3 RAF weight is “offset inside the stage”: the prover uses `γ^4 * raf_shift_claim`
///   in the Stage3 per-stage claim, then the stage itself is folded with an outer factor `γ^2`,
///   yielding the advertised `γ^6` overall.
#[derive(Allocative)]
pub struct BytecodeReadRafSumcheckProver<F: JoltField> {
    /// Per-stage address MLEs F_i(k) built from eq(r_cycle_stage_i, (chunk_index, j)),
    /// bound low-to-high during the address-binding phase.
    F: [MultilinearPolynomial<F>; N_STAGES],
    /// Chunked RA polynomials over address variables (one per dimension `d`), used to form
    /// the product ∏_i ra_i during the cycle-binding phase.
    ra: Vec<RaPolynomial<u8, F>>,
    /// Binding challenges for the first log_K variables of the sumcheck
    r_address_prime: Vec<F::Challenge>,
    /// Per-stage Gruen-split eq polynomials over cycle vars (low-to-high binding order).
    gruen_eq_polys: [GruenSplitEqPolynomial<F>; N_STAGES],
    /// Previous-round claims s_i(0)+s_i(1) per stage, needed for degree-(d+1) univariate recovery.
    prev_round_claims: [F; N_STAGES],
    /// Round polynomials per stage for advancing to the next claim at r_j.
    prev_round_polys: Option<[UniPoly<F>; N_STAGES]>,
    /// Final sumcheck claims of stage Val polynomials (with RAF Int folded where applicable).
    bound_val_evals: Option<[F; N_STAGES]>,
    /// Trace for computing PCs on the fly in init_log_t_rounds.
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    /// Bytecode preprocessing for computing PCs.
    #[allocative(skip)]
    bytecode_preprocessing: Arc<BytecodePreprocessing>,
    pub params: BytecodeReadRafSumcheckParams<F>,
}

impl<F: JoltField> BytecodeReadRafSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheckProver::initialize")]
    pub fn initialize(
        params: BytecodeReadRafSumcheckParams<F>,
        trace: Arc<Vec<Cycle>>,
        bytecode_preprocessing: Arc<BytecodePreprocessing>,
    ) -> Self {
        let claim_per_stage = [
            params.rv_claims[0] + params.gamma_powers[5] * params.raf_claim,
            params.rv_claims[1],
            params.rv_claims[2] + params.gamma_powers[4] * params.raf_shift_claim,
            params.rv_claims[3],
            params.rv_claims[4],
        ];

        // Two-table split-eq optimization for computing F[stage][k] = Σ_{c: PC(c)=k} eq(r_cycle, c).
        //
        // Double summation pattern:
        //   F[stage][k] = Σ_{c_hi} E_hi[c_hi] × ( Σ_{c_lo : PC(c)=k} E_lo[c_lo] )
        //
        // Inner sum (over c_lo): ADDITIONS ONLY - accumulate E_lo contributions by PC
        // Outer sum (over c_hi): ONE multiplication per touched PC, not per cycle
        //
        // This reduces multiplications from O(T × N_STAGES) to O(touched_PCs × out_len × N_STAGES)
        let T = trace.len();
        let K = params.K;
        let log_T = params.log_T;

        // Optimal split: sqrt(T) for balanced tables
        let lo_bits = log_T / 2;
        let hi_bits = log_T - lo_bits;
        let in_len: usize = 1 << lo_bits; // E_lo size (inner loop)
        let out_len: usize = 1 << hi_bits; // E_hi size (outer loop)

        // Pre-compute E_hi[stage][c_hi] and E_lo[stage][c_lo] for all stages in parallel
        let (E_hi, E_lo): ([Vec<F>; N_STAGES], [Vec<F>; N_STAGES]) = rayon::join(
            || {
                params
                    .r_cycles
                    .each_ref()
                    .map(|r_cycle| EqPolynomial::evals(&r_cycle[..hi_bits]))
            },
            || {
                params
                    .r_cycles
                    .each_ref()
                    .map(|r_cycle| EqPolynomial::evals(&r_cycle[hi_bits..]))
            },
        );

        // Process by c_hi blocks, distributing work evenly among threads
        let num_threads = rayon::current_num_threads();
        let chunk_size = out_len.div_ceil(num_threads);

        // Double summation: outer sum over c_hi, inner sum over c_lo
        let F: [Vec<F>; N_STAGES] = E_hi[0]
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                // Per-thread accumulators for final F
                let mut partial: [Vec<F>; N_STAGES] =
                    array::from_fn(|_| unsafe_allocate_zero_vec(K));

                // Per-c_hi inner accumulators (reused across c_hi iterations)
                let mut inner: [Vec<F>; N_STAGES] = array::from_fn(|_| unsafe_allocate_zero_vec(K));

                // Track which PCs were touched in this c_hi block
                let mut touched = Vec::with_capacity(in_len);

                let chunk_start = chunk_idx * chunk_size;
                for (local_idx, _) in chunk.iter().enumerate() {
                    let c_hi = chunk_start + local_idx;
                    let c_hi_base = c_hi * in_len;

                    // Clear inner accumulators for touched PCs only
                    for &k in &touched {
                        for stage in 0..N_STAGES {
                            inner[stage][k] = F::zero();
                        }
                    }
                    touched.clear();

                    // INNER SUM: accumulate E_lo by PC (ADDITIONS ONLY, no multiplications)
                    for c_lo in 0..in_len {
                        let c = c_hi_base + c_lo;
                        if c >= T {
                            break;
                        }

                        let pc = bytecode_preprocessing.get_pc(&trace[c]);

                        // Track touched PCs (avoid duplicates with a simple check)
                        if inner[0][pc].is_zero() {
                            touched.push(pc);
                        }

                        // Accumulate E_lo contributions (addition only!)
                        for stage in 0..N_STAGES {
                            inner[stage][pc] += E_lo[stage][c_lo];
                        }
                    }

                    // OUTER SUM: multiply by E_hi and add to partial (sparse)
                    for &k in &touched {
                        for stage in 0..N_STAGES {
                            partial[stage][k] += E_hi[stage][c_hi] * inner[stage][k];
                        }
                    }
                }

                partial
            })
            .reduce(
                || array::from_fn(|_| unsafe_allocate_zero_vec(K)),
                |mut a, b| {
                    for stage in 0..N_STAGES {
                        a[stage]
                            .par_iter_mut()
                            .zip(b[stage].par_iter())
                            .for_each(|(a, b)| *a += *b);
                    }
                    a
                },
            );

        #[cfg(test)]
        {
            // Verify that for each stage i: sum(val_i[k] * F_i[k] * eq_i[k]) = rv_claim_i
            for i in 0..N_STAGES {
                let computed_claim: F = (0..params.K)
                    .into_par_iter()
                    .map(|k| {
                        let val_k = params.val_polys[i].get_bound_coeff(k);
                        let F_k = F[i][k];
                        val_k * F_k
                    })
                    .sum();
                assert_eq!(
                    computed_claim,
                    params.rv_claims[i],
                    "Stage {} mismatch: computed {} != expected {}",
                    i + 1,
                    computed_claim,
                    params.rv_claims[i]
                );
            }
        }

        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            // Print F_s values and recheck claims against rv_claims
            for stage in 0..N_STAGES {
                let mut computed_claim = F::zero();
                for k in 0..std::cmp::min(params.K, 5) {
                    let mut f_bytes = [0u8; 32];
                    F[stage][k].serialize_compressed(&mut f_bytes[..]).ok();
                    eprint!("[JOLT_F_S] F[{}][{}]_LE=[", stage, k);
                    for b in &f_bytes[..8] { eprint!("{:02x}", b); }
                    eprintln!("]");
                }
                for k in 0..params.K {
                    let val_k = params.val_polys[stage].get_bound_coeff(k);
                    let f_k = F[stage][k];
                    computed_claim += val_k * f_k;
                }
                let mut claim_bytes = [0u8; 32];
                computed_claim.serialize_compressed(&mut claim_bytes[..]).ok();
                let mut rv_bytes = [0u8; 32];
                params.rv_claims[stage].serialize_compressed(&mut rv_bytes[..]).ok();
                eprint!("[JOLT_F_S] stage[{}] Σval*F_LE=[", stage);
                for b in &claim_bytes[..8] { eprint!("{:02x}", b); }
                eprint!("] rv_claim_LE=[");
                for b in &rv_bytes[..8] { eprint!("{:02x}", b); }
                eprintln!("] match={}", computed_claim == params.rv_claims[stage]);
            }
        }

        let F = F.map(MultilinearPolynomial::from);

        let gruen_eq_polys = params
            .r_cycles
            .each_ref()
            .map(|r_cycle| GruenSplitEqPolynomial::new(r_cycle, BindingOrder::LowToHigh));

        Self {
            F,
            ra: Vec::with_capacity(params.d),
            r_address_prime: Vec::with_capacity(params.log_K),
            gruen_eq_polys,
            prev_round_claims: claim_per_stage,
            prev_round_polys: None,
            bound_val_evals: None,
            trace,
            bytecode_preprocessing,
            params,
        }
    }

    fn init_log_t_rounds(&mut self) {
        let int_poly = self.params.int_poly.final_sumcheck_claim();

        // We have a separate Val polynomial for each stage
        // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
        // So we would have:
        // Stage 1: gamma^0 * (Val_1 + gamma^5 * Int)
        // Stage 2: gamma^1 * (Val_2)
        // Stage 3: gamma^2 * (Val_3 + gamma^4 * Int)
        // Stage 4: gamma^3 * (Val_4)
        // Stage 5: gamma^4 * (Val_5)
        // Which matches with the input claim:
        // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * rv_4 + gamma^4 * rv_5 + gamma^5 * raf_1 + gamma^6 * raf_3
        self.bound_val_evals = Some(
            self.params
                .val_polys
                .iter()
                .zip([
                    int_poly * self.params.gamma_powers[5],
                    F::zero(),
                    int_poly * self.params.gamma_powers[4],
                    F::zero(),
                    F::zero(),
                ])
                .map(|(poly, int_poly)| poly.final_sumcheck_claim() + int_poly)
                .collect::<Vec<F>>()
                .try_into()
                .unwrap(),
        );

        // Reverse r_address_prime to get the correct order (it was built low-to-high)
        let mut r_address = std::mem::take(&mut self.r_address_prime);
        r_address.reverse();

        // Drop log_K phase data that's no longer needed (val_polys reduced to bound_val_evals)
        // F polynomials are fully bound and can be dropped
        self.F = array::from_fn(|_| MultilinearPolynomial::default());
        // val_polys are reduced to scalars in bound_val_evals
        self.params.val_polys = array::from_fn(|_| MultilinearPolynomial::default());
        // int_poly is reduced to a scalar
        self.params.int_poly = IdentityPolynomial::new(0);

        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&r_address);

        // Build RA polynomials by iterating over trace and computing PCs on the fly
        self.ra = r_address_chunks
            .iter()
            .enumerate()
            .map(|(i, r_address_chunk)| {
                let ra_i: Vec<Option<u8>> = self
                    .trace
                    .par_iter()
                    .map(|cycle| {
                        let pc = self.bytecode_preprocessing.get_pc(cycle);
                        Some(self.params.one_hot_params.bytecode_pc_chunk(pc, i))
                    })
                    .collect();
                RaPolynomial::new(Arc::new(ra_i), EqPolynomial::evals(r_address_chunk))
            })
            .collect();

        // Drop trace and preprocessing - no longer needed after this
        self.trace = Arc::new(Vec::new());
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for BytecodeReadRafSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheckProver::compute_message")]
    fn compute_message(&mut self, round: usize, _previous_claim: F) -> UniPoly<F> {
        if round < self.params.log_K {
            const DEGREE: usize = 2;

            // Evaluation at [0, 2] for each stage.
            let eval_per_stage: [[F; DEGREE]; N_STAGES] = (0..self.params.val_polys[0].len() / 2)
                .into_par_iter()
                .map(|i| {
                    let ra_evals = self.F.each_ref().map(|poly| {
                        poly.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh)
                    });

                    let int_evals =
                        self.params.int_poly
                            .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                    // We have a separate Val polynomial for each stage
                    // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
                    // So we would have:
                    // Stage 1: Val_1 + gamma^5 * Int
                    // Stage 2: Val_2
                    // Stage 3: Val_3 + gamma^4 * Int
                    // Stage 4: Val_4
                    // Stage 5: Val_5
                    // Which matches with the input claim:
                    // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * rv_4 + gamma^4 * rv_5 + gamma^5 * raf_1 + gamma^6 * raf_3
                    let mut val_evals = self
                        .params.val_polys
                        .iter()
                        // Val polynomials
                        .map(|val| val.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh))
                        // Here are the RAF polynomials and their powers
                        .zip([Some(&int_evals), None, Some(&int_evals), None, None])
                        .zip([Some(self.params.gamma_powers[5]), None, Some(self.params.gamma_powers[4]), None, None])
                        .map(|((val_evals, int_evals), gamma)| {
                            std::array::from_fn::<F, DEGREE, _>(|j| {
                                val_evals[j]
                                    + int_evals.map_or(F::zero(), |int_evals| {
                                        int_evals[j] * gamma.unwrap()
                                    })
                            })
                        });

                    array::from_fn(|stage| {
                        let [ra_at_0, ra_at_2] = ra_evals[stage];
                        let [val_at_0, val_at_2] = val_evals.next().unwrap();
                        [ra_at_0 * val_at_0, ra_at_2 * val_at_2]
                    })
                })
                .reduce(
                    || [[F::zero(); DEGREE]; N_STAGES],
                    |a, b| array::from_fn(|i| array::from_fn(|j| a[i][j] + b[i][j])),
                );

            let mut round_polys: [_; N_STAGES] = array::from_fn(|_| UniPoly::zero());
            let mut agg_round_poly = UniPoly::zero();

            for (stage, evals) in eval_per_stage.into_iter().enumerate() {
                let [eval_at_0, eval_at_2] = evals;
                let eval_at_1 = self.prev_round_claims[stage] - eval_at_0;
                let round_poly = UniPoly::from_evals(&[eval_at_0, eval_at_1, eval_at_2]);
                agg_round_poly += &(&round_poly * self.params.gamma_powers[stage]);
                round_polys[stage] = round_poly;
            }

            self.prev_round_polys = Some(round_polys);

            agg_round_poly
        } else {
            let degree = <Self as SumcheckInstanceProver<F, T>>::degree(self);

            let out_len = self.gruen_eq_polys[0].E_out_current().len();
            let in_len = self.gruen_eq_polys[0].E_in_current().len();
            let in_n_vars = in_len.log_2();

            // Evaluations on [1, ..., degree - 2, inf] (for each stage).
            let mut evals_per_stage: [Vec<F>; N_STAGES] = (0..out_len)
                .into_par_iter()
                .map(|j_hi| {
                    let mut ra_eval_pairs = vec![(F::zero(), F::zero()); self.ra.len()];
                    let mut ra_prod_evals = vec![F::zero(); degree - 1];
                    let mut evals_per_stage: [_; N_STAGES] =
                        array::from_fn(|_| vec![F::Unreduced::zero(); degree - 1]);

                    for j_lo in 0..in_len {
                        let j = j_lo + (j_hi << in_n_vars);

                        for (i, ra_i) in self.ra.iter().enumerate() {
                            let ra_i_eval_at_j_0 = ra_i.get_bound_coeff(j * 2);
                            let ra_i_eval_at_j_1 = ra_i.get_bound_coeff(j * 2 + 1);
                            ra_eval_pairs[i] = (ra_i_eval_at_j_0, ra_i_eval_at_j_1);
                        }
                        // Eval prod_i ra_i(x).
                        eval_linear_prod_assign(&ra_eval_pairs, &mut ra_prod_evals);

                        for stage in 0..N_STAGES {
                            let eq_in_eval = self.gruen_eq_polys[stage].E_in_current()[j_lo];
                            for i in 0..degree - 1 {
                                evals_per_stage[stage][i] +=
                                    eq_in_eval.mul_unreduced::<9>(ra_prod_evals[i]);
                            }
                        }
                    }

                    array::from_fn(|stage| {
                        let eq_out_eval = self.gruen_eq_polys[stage].E_out_current()[j_hi];
                        evals_per_stage[stage]
                            .iter()
                            .map(|v| eq_out_eval * F::from_montgomery_reduce(*v))
                            .collect()
                    })
                })
                .reduce(
                    || array::from_fn(|_| vec![F::zero(); degree - 1]),
                    |a, b| array::from_fn(|i| zip_eq(&a[i], &b[i]).map(|(a, b)| *a + *b).collect()),
                );
            // Multiply by bound values.
            let bound_val_evals = self.bound_val_evals.as_ref().unwrap();
            for (stage, evals) in evals_per_stage.iter_mut().enumerate() {
                evals.iter_mut().for_each(|v| *v *= bound_val_evals[stage]);
            }

            let mut round_polys: [_; N_STAGES] = array::from_fn(|_| UniPoly::zero());
            let mut agg_round_poly = UniPoly::zero();

            // Obtain round poly for each stage and perform RLC.
            for (stage, evals) in evals_per_stage.iter().enumerate() {
                let claim = self.prev_round_claims[stage];
                let round_poly = self.gruen_eq_polys[stage].gruen_poly_from_evals(evals, claim);
                agg_round_poly += &(&round_poly * self.params.gamma_powers[stage]);
                round_polys[stage] = round_poly;
            }

            self.prev_round_polys = Some(round_polys);

            agg_round_poly
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if let Some(prev_round_polys) = self.prev_round_polys.take() {
            self.prev_round_claims = prev_round_polys.map(|poly| poly.evaluate(&r_j));
        }

        if round < self.params.log_K {
            self.params
                .val_polys
                .iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
            self.params
                .int_poly
                .bind_parallel(r_j, BindingOrder::LowToHigh);
            self.F
                .iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
            self.r_address_prime.push(r_j);
            if round == self.params.log_K - 1 {
                self.init_log_t_rounds();
            }
        } else {
            self.ra
                .iter_mut()
                .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh));
            self.gruen_eq_polys
                .iter_mut()
                .for_each(|poly| poly.bind(r_j));
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address, r_cycle) = opening_point.split_at(self.params.log_K);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&r_address.r);

        for i in 0..self.params.d {
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                r_address_chunks[i].clone(),
                r_cycle.clone().into(),
                vec![self.ra[i].final_sumcheck_claim()],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct BytecodeReadRafSumcheckVerifier<F: JoltField> {
    params: BytecodeReadRafSumcheckParams<F>,
}

impl<F: JoltField> BytecodeReadRafSumcheckVerifier<F> {
    pub fn gen(
        bytecode_preprocessing: &BytecodePreprocessing,
        n_cycle_vars: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        Self {
            params: BytecodeReadRafSumcheckParams::gen(
                bytecode_preprocessing,
                n_cycle_vars,
                one_hot_params,
                opening_accumulator,
                transcript,
            ),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for BytecodeReadRafSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address_prime, r_cycle_prime) = opening_point.split_at(self.params.log_K);
        // r_cycle is bound LowToHigh, so reverse

        // Debug: print r_address_prime and r_cycle_prime
        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            eprintln!("[BCRAF_VERIFY] r_address_prime (len={}):", r_address_prime.r.len());
            for (i, v) in r_address_prime.r.iter().enumerate() {
                let mut b = [0u8; 32]; v.serialize_compressed(&mut b[..]).ok();
                let h: String = b.iter().map(|b| format!("{:02x}", b)).collect();
                eprintln!("  r_addr[{}]_LE=[{}]", i, h);
            }
            eprintln!("[BCRAF_VERIFY] r_cycle_prime (len={}):", r_cycle_prime.r.len());
            for (i, v) in r_cycle_prime.r.iter().enumerate() {
                let mut b = [0u8; 32]; v.serialize_compressed(&mut b[..]).ok();
                let h: String = b.iter().map(|b| format!("{:02x}", b)).collect();
                eprintln!("  r_cyc[{}]_LE=[{}]", i, h);
            }
        }

        // Print the sumcheck challenges used for address (LH order)
        {
            use ark_serialize::CanonicalSerialize;
            eprintln!("[BCRAF_VERIFY] sumcheck_challenges[..log_K] (len={}, LH order):", self.params.log_K);
            for (i, c) in sumcheck_challenges[..self.params.log_K].iter().enumerate() {
                let c_f: F = (*c).into();
                let mut cb = [0u8; 32];
                c_f.serialize_compressed(&mut cb[..]).ok();
                let ch: String = cb.iter().map(|b| format!("{:02x}", b)).collect();
                eprintln!("  ch[{}]_LE=[{}]", i, ch);
            }
            eprintln!("[BCRAF_VERIFY] r_address_prime.r (len={}, after normalize/reverse):", r_address_prime.r.len());
            for (i, r) in r_address_prime.r.iter().enumerate() {
                let mut rb = [0u8; 32];
                r.serialize_compressed(&mut rb[..]).ok();
                let rh: String = rb.iter().map(|b| format!("{:02x}", b)).collect();
                eprintln!("  r_addr[{}]_LE=[{}]", i, rh);
            }
        }
        let int_poly = self.params.int_poly.evaluate(&r_address_prime.r);

        // ALWAYS-ON: print int_poly evaluation and RAF terms
        {
            use ark_serialize::CanonicalSerialize;
            let mut ipb = [0u8; 32]; int_poly.serialize_compressed(&mut ipb[..]).ok();
            let iph: String = ipb.iter().map(|b| format!("{:02x}", b)).collect();
            eprintln!("[BCRAF_VERIFY] int_poly_eval_LE=[{}]", iph);

            let raf0 = int_poly * self.params.gamma_powers[5];
            let raf2 = int_poly * self.params.gamma_powers[4];
            let mut r0b = [0u8; 32]; raf0.serialize_compressed(&mut r0b[..]).ok();
            let mut r2b = [0u8; 32]; raf2.serialize_compressed(&mut r2b[..]).ok();
            let r0h: String = r0b.iter().map(|b| format!("{:02x}", b)).collect();
            let r2h: String = r2b.iter().map(|b| format!("{:02x}", b)).collect();
            eprintln!("[BCRAF_VERIFY] gamma5*int_LE=[{}]", r0h);
            eprintln!("[BCRAF_VERIFY] gamma4*int_LE=[{}]", r2h);
        }

        let ra_claims = (0..self.params.d).map(|i| {
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeRa(i),
                    SumcheckId::BytecodeReadRaf,
                )
                .1
        });

        // We have a separate Val polynomial for each stage
        // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
        // So we would have:
        // Stage 1: gamma^0 * (Val_1 + gamma^5 * Int)
        // Stage 2: gamma^1 * (Val_2)
        // Stage 3: gamma^2 * (Val_3 + gamma^4 * Int)
        // Stage 4: gamma^3 * (Val_4)
        // Stage 5: gamma^4 * (Val_5)
        // Which matches with the input claim:
        // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * rv_4 + gamma^4 * rv_5 + gamma^5 * raf_1 + gamma^6 * raf_3
        let raf_terms = [
            int_poly * self.params.gamma_powers[5], // RAF for Stage1
            F::zero(),                              // There's no raf for Stage2
            int_poly * self.params.gamma_powers[4], // RAF for Stage3
            F::zero(),                              // There's no raf for Stage4
            F::zero(),                              // There's no raf for Stage5
        ];

        let val = self
            .params
            .val_polys
            .iter()
            .zip(&self.params.r_cycles)
            .zip(&self.params.gamma_powers)
            .zip(raf_terms)
            .enumerate()
            .map(|(s, (((val, r_cycle), gamma), raf))| {
                // Print first few val_poly coefficients for comparison with Zolt prover
                {
                    use ark_serialize::CanonicalSerialize;
                    let n_entries = val.len().min(32);
                    for k in 0..n_entries {
                        let coeff = val.get_coeff(k);
                        let mut cb = [0u8; 32];
                        coeff.serialize_compressed(&mut cb[..]).ok();
                        eprintln!("[JOLT_VP] Val[{}][{}]_LE=[{:02x},{:02x},{:02x},{:02x},{:02x},{:02x},{:02x},{:02x}]",
                            s, k, cb[0], cb[1], cb[2], cb[3], cb[4], cb[5], cb[6], cb[7]);
                    }
                }
                let val_eval = val.evaluate(&r_address_prime.r);
                // Cross-check: compute val_eval manually using LE eq (Zolt's convention)
                let lh_val_eval = {
                    let r_lh: Vec<F> = sumcheck_challenges[..self.params.log_K].iter().map(|c| (*c).into()).collect();
                    // Use little-endian eq computation: r[0] = LSB
                    let n = r_lh.len();
                    let size = 1usize << n;
                    let mut eq = vec![F::one(); size];
                    for i in 0..n {
                        let r_i = r_lh[i];
                        let one_minus_r = F::one() - r_i;
                        let cur_size = 1usize << i;
                        for j in (0..cur_size).rev() {
                            eq[j + cur_size] = eq[j] * r_i;
                            eq[j] = eq[j] * one_minus_r;
                        }
                    }
                    // Print eq table entries for comparison with Zolt
                    if s == 0 {
                        use ark_serialize::CanonicalSerialize;
                        for k in 0..size {
                            let mut eb = [0u8; 32];
                            eq[k].serialize_compressed(&mut eb[..]).ok();
                            let eh: String = eb.iter().map(|b| format!("{:02x}", b)).collect();
                            eprintln!("[JOLT_EQ_LH] eq[{}]_LE=[{}]", k, eh);
                        }
                    }
                    let mut sum = F::zero();
                    for k in 0..val.len() {
                        let prod = val.get_coeff(k) * eq[k];
                        sum += prod;
                        if s == 0 {
                            use ark_serialize::CanonicalSerialize;
                            let mut pb = [0u8; 32]; prod.serialize_compressed(&mut pb[..]).ok();
                            let mut sb = [0u8; 32]; sum.serialize_compressed(&mut sb[..]).ok();
                            let phs: String = pb[..8].iter().map(|b| format!("{:02x}", b)).collect();
                            let shs: String = sb[..8].iter().map(|b| format!("{:02x}", b)).collect();
                            eprintln!("[JOLT_PARTIAL_SUM] k={}: prod_LE=[{}] sum_LE=[{}]", k, phs, shs);
                        }
                    }
                    sum
                };
                let eq_eval = EqPolynomial::<F>::mle(r_cycle, &r_cycle_prime.r);
                let stage_contrib = (val_eval + raf) * eq_eval * gamma;
                let bound_val = (val_eval + raf) * gamma;
                {
                    use ark_serialize::CanonicalSerialize;
                    let mut vb = [0u8; 32]; val_eval.serialize_compressed(&mut vb[..]).ok();
                    let mut lb = [0u8; 32]; lh_val_eval.serialize_compressed(&mut lb[..]).ok();
                    let mut eb = [0u8; 32]; eq_eval.serialize_compressed(&mut eb[..]).ok();
                    let mut bb = [0u8; 32]; bound_val.serialize_compressed(&mut bb[..]).ok();
                    let vh: String = vb.iter().map(|b| format!("{:02x}", b)).collect();
                    let lh: String = lb.iter().map(|b| format!("{:02x}", b)).collect();
                    let eh: String = eb.iter().map(|b| format!("{:02x}", b)).collect();
                    let bh: String = bb.iter().map(|b| format!("{:02x}", b)).collect();
                    eprintln!("[BCRAF_VERIFY] stage[{}]: val_eval_LE=[{}]", s, vh);
                    eprintln!("[BCRAF_VERIFY] stage[{}]: lh_eval_LE=[{}] match_val={}", s, lh, lh_val_eval == val_eval);
                    eprintln!("[BCRAF_VERIFY] stage[{}]: eq_LE=[{}] bound_val_LE=[{}]", s, eh, bh);
                }
                stage_contrib
            })
            .sum::<F>();

        {
            use ark_serialize::CanonicalSerialize;
            let mut vb = [0u8; 32];
            val.serialize_compressed(&mut vb[..]).ok();
            let vh: String = vb.iter().map(|b| format!("{:02x}", b)).collect();
            eprintln!("[BCRAF_VERIFY] val_sum_LE=[{}]", vh);
        }
        eprintln!("[BCRAF_VERIFY] val (sum) = {:?}", val);

        let result = ra_claims.enumerate().fold(val, |running, (i, ra_claim)| {
            use ark_serialize::CanonicalSerialize;
            let mut rab = [0u8; 32];
            ra_claim.serialize_compressed(&mut rab[..]).ok();
            let rah: String = rab.iter().map(|b| format!("{:02x}", b)).collect();
            eprintln!("[BCRAF_VERIFY] ra[{}]_LE=[{}]", i, rah);
            running * ra_claim
        });

        {
            use ark_serialize::CanonicalSerialize;
            let mut rb = [0u8; 32];
            result.serialize_compressed(&mut rb[..]).ok();
            let rh: String = rb.iter().map(|b| format!("{:02x}", b)).collect();
            eprintln!("[BCRAF_VERIFY] expected_output_LE=[{}]", rh);
        }
        eprintln!("[BCRAF_VERIFY] expected_output = {:?}", result);

        result
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address, r_cycle) = opening_point.split_at(self.params.log_K);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&r_address.r);

        (0..self.params.d).for_each(|i| {
            let opening_point = [&r_address_chunks[i][..], &r_cycle.r].concat();
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                opening_point,
            );
        });
    }
}

#[derive(Allocative, Clone)]
pub struct BytecodeReadRafSumcheckParams<F: JoltField> {
    /// Index `i` stores `gamma^i`.
    pub gamma_powers: Vec<F>,
    /// RLC of stage rv_claims and RAF claims (per Stage1/Stage3) used as the sumcheck LHS.
    pub input_claim: F,
    /// RaParams
    pub one_hot_params: OneHotParams,
    /// Bytecode length.
    pub K: usize,
    /// log2(K) and log2(T) used to determine round counts.
    pub log_K: usize,
    pub log_T: usize,
    /// Number of address chunks (and RA polynomials in the product).
    pub d: usize,
    /// Stage Val polynomials evaluated over address vars.
    pub val_polys: [MultilinearPolynomial<F>; N_STAGES],
    /// Stage rv claims.
    pub rv_claims: [F; N_STAGES],
    pub raf_claim: F,
    pub raf_shift_claim: F,
    /// Identity polynomial over address vars used to inject RAF contributions.
    pub int_poly: IdentityPolynomial<F>,
    pub r_cycles: [Vec<F::Challenge>; N_STAGES],
}

impl<F: JoltField> BytecodeReadRafSumcheckParams<F> {
    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheckParams::gen")]
    pub fn gen(
        bytecode_preprocessing: &BytecodePreprocessing,
        n_cycle_vars: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma_powers: Vec<F> = transcript.challenge_scalar_powers(7);

        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            let mut g_bytes = vec![];
            gamma_powers[1].serialize_compressed(&mut g_bytes).unwrap();
            eprint!("[JOLT STAGE6] bytecodeRaf_gamma = [");
            for i in 0..8 { eprint!("{:02x},", g_bytes[i]); }
            eprintln!("]");
        }

        let bytecode = &bytecode_preprocessing.bytecode;

        // Generate all stage-specific gamma powers upfront (order must match verifier)
        let stage1_gammas: Vec<F> = transcript.challenge_scalar_powers(2 + NUM_CIRCUIT_FLAGS);
        let stage2_gammas: Vec<F> = transcript.challenge_scalar_powers(4);
        let stage3_gammas: Vec<F> = transcript.challenge_scalar_powers(9);
        let stage4_gammas: Vec<F> = transcript.challenge_scalar_powers(3);
        let stage5_gammas: Vec<F> = transcript.challenge_scalar_powers(2 + NUM_LOOKUP_TABLES);

        // Compute rv_claims (these don't iterate bytecode, just query opening accumulator)
        let rv_claim_1 = Self::compute_rv_claim_1(opening_accumulator, &stage1_gammas);
        let rv_claim_2 = Self::compute_rv_claim_2(opening_accumulator, &stage2_gammas);
        let rv_claim_3 = Self::compute_rv_claim_3(opening_accumulator, &stage3_gammas);
        let rv_claim_4 = Self::compute_rv_claim_4(opening_accumulator, &stage4_gammas);
        let rv_claim_5 = Self::compute_rv_claim_5(opening_accumulator, &stage5_gammas);
        let rv_claims = [rv_claim_1, rv_claim_2, rv_claim_3, rv_claim_4, rv_claim_5];

        #[cfg(feature = "zolt-debug")]
        {
            for (i, rv) in rv_claims.iter().enumerate() {
                use ark_serialize::CanonicalSerialize;
                let mut bytes = [0u8; 32];
                rv.serialize_compressed(&mut bytes[..]).ok();
                eprintln!("[JOLT_BCRAF] rv_claim[{}]_LE=[{:02x},{:02x},{:02x},{:02x},{:02x},{:02x},{:02x},{:02x}]",
                    i, bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]);
            }
        }

        // Pre-compute eq_r_register for stages 4 and 5 (they use different r_register points)
        let r_register_4 = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa,
                SumcheckId::RegistersReadWriteChecking,
            )
            .0
            .r;
        let eq_r_register_4 =
            EqPolynomial::<F>::evals(&r_register_4[..(REGISTER_COUNT as usize).log_2()]);

        let r_register_5 = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa,
                SumcheckId::RegistersValEvaluation,
            )
            .0
            .r;
        let eq_r_register_5 =
            EqPolynomial::<F>::evals(&r_register_5[..(REGISTER_COUNT as usize).log_2()]);

        // Debug: print r_register values and eq table entries
        #[cfg(feature = "zolt-debug")]
        {
            eprintln!("[JOLT STAGE6] r_register_4 (len={}):", r_register_4.len());
            for (i, rv) in r_register_4.iter().enumerate() {
                use ark_serialize::CanonicalSerialize;
                let mut bytes = [0u8; 32];
                rv.serialize_compressed(&mut bytes[..]).ok();
                eprintln!("  r_register_4[{}] = {:02x?}", i, &bytes);
            }
            eprintln!("[JOLT STAGE6] r_register_5 (len={}):", r_register_5.len());
            for (i, rv) in r_register_5.iter().enumerate() {
                use ark_serialize::CanonicalSerialize;
                let mut bytes = [0u8; 32];
                rv.serialize_compressed(&mut bytes[..]).ok();
                eprintln!("  r_register_5[{}] = {:02x?}", i, &bytes);
            }
            // Print eq_table_4 entries at specific indices in LE hex
            eprintln!("[JOLT STAGE6] eq_r_register_4 (len={}):", eq_r_register_4.len());
            for idx in [0usize, 1, 2, 8, 10, 15, 31, 127] {
                if idx < eq_r_register_4.len() {
                    use ark_serialize::CanonicalSerialize;
                    let mut bytes = [0u8; 32];
                    eq_r_register_4[idx].serialize_compressed(&mut bytes[..]).ok();
                    let hex_str: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
                    eprintln!("  eq4[{}]_LE=[{}]", idx, hex_str);
                }
            }
            // Print stage4_gammas in LE hex
            eprintln!("[JOLT STAGE6] stage4_gammas:");
            for i in 0..3 {
                use ark_serialize::CanonicalSerialize;
                let mut bytes = [0u8; 32];
                stage4_gammas[i].serialize_compressed(&mut bytes[..]).ok();
                let hex_str: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
                eprintln!("  gamma4[{}]_LE=[{}]", i, hex_str);
            }
        }

        // Compute val polynomials. When Zolt raw words are available, use the Zolt-specific
        // code path that properly handles termination entries (LUI/ADDI/SB at termination_base_pc).
        // The vanilla compute_val_polys would treat those entries as NoOp since the bytecode
        // array only contains Instruction::NoOp at those indices.
        #[cfg(feature = "zolt-debug")]
        let val_polys = {
            let raw_words_opt = super::ZOLT_RAW_WORDS.get();
            let term_addr_opt = super::ZOLT_TERMINATION_ADDRESS.get();
            if let (Some(raw_words), Some(&term_addr)) = (raw_words_opt, term_addr_opt) {
                eprintln!("[BCRAF] Using compute_val_polys_zolt with {} raw words, termination_addr=0x{:x}", raw_words.len(), term_addr);
                let zolt_flags = ZoltBytecodeFlags::from_raw_words(raw_words, bytecode, term_addr);
                // Pad to bytecode_K if needed
                let bytecode_k = bytecode.len();
                let mut padded_flags = zolt_flags;
                while padded_flags.len() < bytecode_k {
                    padded_flags.push(ZoltBytecodeFlags::noop());
                }
                padded_flags.truncate(bytecode_k);
                Self::compute_val_polys_zolt(
                    &padded_flags,
                    &eq_r_register_4,
                    &eq_r_register_5,
                    &stage1_gammas,
                    &stage2_gammas,
                    &stage3_gammas,
                    &stage4_gammas,
                    &stage5_gammas,
                )
            } else {
                eprintln!("[BCRAF] Using vanilla compute_val_polys (no Zolt raw words)");
                Self::compute_val_polys(
                    bytecode,
                    &eq_r_register_4,
                    &eq_r_register_5,
                    &stage1_gammas,
                    &stage2_gammas,
                    &stage3_gammas,
                    &stage4_gammas,
                    &stage5_gammas,
                )
            }
        };
        #[cfg(not(feature = "zolt-debug"))]
        let val_polys = Self::compute_val_polys(
            bytecode,
            &eq_r_register_4,
            &eq_r_register_5,
            &stage1_gammas,
            &stage2_gammas,
            &stage3_gammas,
            &stage4_gammas,
            &stage5_gammas,
        );

        // Debug: print all val poly values for comparison with Zolt
        #[cfg(feature = "zolt-debug")]
        {
            let K = bytecode.len();
            for s in 0..5 {
                for k in 0..K {
                    use ark_serialize::CanonicalSerialize;
                    let mut bytes = [0u8; 32];
                    val_polys[s].get_coeff(k).serialize_compressed(&mut bytes[..]).ok();
                    let hex_str: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
                    eprintln!("[JOLT_VAL_POLY] Val[{}][{}]_LE=[{}]", s, k, hex_str);
                }
            }
        }

        let int_poly = IdentityPolynomial::new(one_hot_params.bytecode_k.log_2());

        let (_, raf_claim) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        let (_, raf_shift_claim) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            let mut bytes = [0u8; 32];
            raf_claim.serialize_compressed(&mut bytes[..]).ok();
            eprintln!("[JOLT_BCRAF] raf_claim_LE=[{:02x},{:02x},{:02x},{:02x},{:02x},{:02x},{:02x},{:02x}]",
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]);
            raf_shift_claim.serialize_compressed(&mut bytes[..]).ok();
            eprintln!("[JOLT_BCRAF] raf_shift_claim_LE=[{:02x},{:02x},{:02x},{:02x},{:02x},{:02x},{:02x},{:02x}]",
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]);
            // Per-stage claims with RAF folded in
            let cps0 = rv_claim_1 + gamma_powers[5] * raf_claim;
            let cps2 = rv_claim_3 + gamma_powers[4] * raf_shift_claim;
            cps0.serialize_compressed(&mut bytes[..]).ok();
            eprintln!("[JOLT_BCRAF] claim_per_stage[0]_LE=[{:02x},{:02x},{:02x},{:02x},{:02x},{:02x},{:02x},{:02x}]",
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]);
            cps2.serialize_compressed(&mut bytes[..]).ok();
            eprintln!("[JOLT_BCRAF] claim_per_stage[2]_LE=[{:02x},{:02x},{:02x},{:02x},{:02x},{:02x},{:02x},{:02x}]",
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]);
        }

        let input_claim = [
            rv_claim_1,
            rv_claim_2,
            rv_claim_3,
            rv_claim_4,
            rv_claim_5,
            raf_claim,
            raf_shift_claim,
        ]
        .iter()
        .zip(&gamma_powers)
        .map(|(claim, g)| *claim * g)
        .sum();

        let (r_cycle_1, _) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
        let (r_cycle_2, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::Jump),
            SumcheckId::SpartanProductVirtualization,
        );
        let (r_cycle_3, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, r_cycle_4) = r.split_at((REGISTER_COUNT as usize).log_2());
        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, r_cycle_5) = r.split_at((REGISTER_COUNT as usize).log_2());
        let r_cycles = [
            r_cycle_1.r,
            r_cycle_2.r,
            r_cycle_3.r,
            r_cycle_4.r,
            r_cycle_5.r,
        ];

        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            for (stage, rc) in r_cycles.iter().enumerate() {
                eprint!("[JOLT_BCRAF] r_cycle[{}] (len={}):", stage, rc.len());
                for (i, v) in rc.iter().enumerate() {
                    // Convert Challenge to F first, then serialize
                    let f_val: F = (*v).into();
                    let mut bytes = [0u8; 32];
                    f_val.serialize_compressed(&mut bytes[..]).ok();
                    eprint!(" [{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}]",
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]);
                    if i >= 3 { eprint!("..."); break; }
                }
                eprintln!();
            }
        }

        // Note: We don't have r_address at this point (it comes from sumcheck_challenges),
        // so we initialize r_address_chunks as empty and will compute it later
        Self {
            gamma_powers,
            input_claim,
            one_hot_params: one_hot_params.clone(),
            K: one_hot_params.bytecode_k,
            log_K: one_hot_params.bytecode_k.log_2(),
            d: one_hot_params.bytecode_d,
            log_T: n_cycle_vars,
            val_polys,
            rv_claims,
            raf_claim,
            raf_shift_claim,
            int_poly,
            r_cycles,
        }
    }

    /// Fused computation of all Val polynomials in a single parallel pass over bytecode.
    ///
    /// This computes all 5 stage-specific Val(k) polynomials simultaneously, avoiding
    /// 5 separate passes through the bytecode. Each stage has its own gamma powers
    /// and formula for Val(k).
    #[allow(clippy::too_many_arguments)]
    fn compute_val_polys(
        bytecode: &[Instruction],
        eq_r_register_4: &[F],
        eq_r_register_5: &[F],
        stage1_gammas: &[F],
        stage2_gammas: &[F],
        stage3_gammas: &[F],
        stage4_gammas: &[F],
        stage5_gammas: &[F],
    ) -> [MultilinearPolynomial<F>; N_STAGES] {
        let K = bytecode.len();

        // Pre-allocate output vectors for each stage
        let mut vals: [Vec<F>; N_STAGES] = array::from_fn(|_| unsafe_allocate_zero_vec(K));
        let [v0, v1, v2, v3, v4] = &mut vals;

        // Fused parallel iteration: compute all 5 val entries for each instruction
        bytecode
            .par_iter()
            .zip(v0.par_iter_mut())
            .zip(v1.par_iter_mut())
            .zip(v2.par_iter_mut())
            .zip(v3.par_iter_mut())
            .zip(v4.par_iter_mut())
            .enumerate()
            .for_each(|(k, (((((instruction, o0), o1), o2), o3), o4))| {
                let instr = instruction.normalize();
                let circuit_flags = instruction.circuit_flags();
                let instr_flags = instruction.instruction_flags();

                #[cfg(feature = "zolt-debug")]
                if k < 20 {
                    let lt = instruction.lookup_table();
                    let lt_idx = lt.as_ref().map(|t| crate::zkvm::lookup_table::LookupTables::enum_index(t));
                    eprintln!(
                        "[VAL_POLY] k={} addr=0x{:08x} rd={:?} rs1={:?} rs2={:?} imm={} cf={:?} if={:?} lt={:?} interleaved={}",
                        k, instr.address, instr.operands.rd, instr.operands.rs1, instr.operands.rs2,
                        instr.operands.imm, &circuit_flags, &instr_flags, lt_idx,
                        circuit_flags.is_interleaved_operands()
                    );
                }

                // Stage 1 (Spartan outer sumcheck)
                // Val(k) = unexpanded_pc(k) + γ·imm(k)
                //          + γ²·circuit_flags[0](k) + γ³·circuit_flags[1](k) + ...
                // This virtualizes claims output by Spartan's "outer" sumcheck.
                {
                    let mut lc = F::from_u64(instr.address as u64);
                    lc += instr.operands.imm.field_mul(stage1_gammas[1]);
                    // sanity check
                    debug_assert!(
                        !circuit_flags[CircuitFlags::IsCompressed]
                            || !circuit_flags[CircuitFlags::DoNotUpdateUnexpandedPC]
                    );
                    for (flag, gamma_power) in circuit_flags.iter().zip(stage1_gammas[2..].iter()) {
                        if *flag {
                            lc += *gamma_power;
                        }
                    }
                    *o0 = lc;
                }

                // Stage 2 (product virtualization, de-duplicated factors)
                // Val(k) = jump_flag(k) + γ·branch_flag(k)
                //          + γ²·is_rd_not_zero_flag(k) + γ³·write_lookup_output_to_rd_flag(k)
                // where jump_flag(k) = 1 if instruction k is a jump, 0 otherwise;
                //       branch_flag(k) = 1 if instruction k is a branch, 0 otherwise;
                //       is_rd_not_zero_flag(k) = 1 if instruction k has rd != 0;
                //       write_lookup_output_to_rd_flag(k) = 1 if instruction k writes lookup output to rd.
                // This Val matches the fused product sumcheck.
                {
                    let mut lc = F::zero();
                    if circuit_flags[CircuitFlags::Jump] {
                        lc += stage2_gammas[0];
                    }
                    if instr_flags[InstructionFlags::Branch] {
                        lc += stage2_gammas[1];
                    }
                    if instr_flags[InstructionFlags::IsRdNotZero] {
                        lc += stage2_gammas[2];
                    }
                    if circuit_flags[CircuitFlags::WriteLookupOutputToRD] {
                        lc += stage2_gammas[3];
                    }
                    *o1 = lc;
                }

                // Stage 3 (Shift sumcheck)
                // Val(k) = imm(k) + γ·unexpanded_pc(k)
                //          + γ²·left_operand_is_rs1_value(k) + γ³·left_operand_is_pc(k)
                //          + γ⁴·right_operand_is_rs2_value(k) + γ⁵·right_operand_is_imm(k)
                //          + γ⁶·is_noop(k) + γ⁷·virtual_instruction(k) + γ⁸·is_first_in_sequence(k)
                // This virtualizes claims output by the ShiftSumcheck.
                {
                    let mut lc = F::from_i128(instr.operands.imm);
                    lc += stage3_gammas[1].mul_u64(instr.address as u64);
                    if instr_flags[InstructionFlags::LeftOperandIsRs1Value] {
                        lc += stage3_gammas[2];
                    }
                    if instr_flags[InstructionFlags::LeftOperandIsPC] {
                        lc += stage3_gammas[3];
                    }
                    if instr_flags[InstructionFlags::RightOperandIsRs2Value] {
                        lc += stage3_gammas[4];
                    }
                    if instr_flags[InstructionFlags::RightOperandIsImm] {
                        lc += stage3_gammas[5];
                    }
                    if instr_flags[InstructionFlags::IsNoop] {
                        lc += stage3_gammas[6];
                    }
                    if circuit_flags[CircuitFlags::VirtualInstruction] {
                        lc += stage3_gammas[7];
                    }
                    if circuit_flags[CircuitFlags::IsFirstInSequence] {
                        lc += stage3_gammas[8];
                    }
                    *o2 = lc;
                }

                // Stage 4 (registers read/write checking sumcheck)
                // Val(k) = eq(rd(k), r_register) + γ·eq(rs1(k), r_register) + γ²·eq(rs2(k), r_register)
                // where rd(k, r) = 1 if the k'th instruction in the bytecode has rd = r,
                // and analogously for rs1(k, r) and rs2(k, r).
                // This virtualizes claims output by the registers read/write checking sumcheck.
                {
                    let rd_eq = instr
                        .operands
                        .rd
                        .map_or(F::zero(), |r| eq_r_register_4[r as usize]);
                    let rs1_eq = instr
                        .operands
                        .rs1
                        .map_or(F::zero(), |r| eq_r_register_4[r as usize]);
                    let rs2_eq = instr
                        .operands
                        .rs2
                        .map_or(F::zero(), |r| eq_r_register_4[r as usize]);
                    *o3 = rd_eq * stage4_gammas[0]
                        + rs1_eq * stage4_gammas[1]
                        + rs2_eq * stage4_gammas[2];
                }

                // Stage 5 (registers val-evaluation + instruction lookups sumcheck)
                // Val(k) = eq(rd(k), r_register) + γ·raf_flag(k)
                //          + γ²·lookup_table_flag[0](k) + γ³·lookup_table_flag[1](k) + ...
                // where rd(k, r) = 1 if the k'th instruction in the bytecode has rd = r,
                // and raf_flag(k) = 1 if instruction k is NOT interleaved operands.
                // This virtualizes the claim output by the registers val-evaluation sumcheck
                // and the instruction lookups sumcheck.
                {
                    let mut lc = instr
                        .operands
                        .rd
                        .map_or(F::zero(), |r| eq_r_register_5[r as usize]);
                    if !circuit_flags.is_interleaved_operands() {
                        lc += stage5_gammas[1];
                    }
                    if let Some(table) = instruction.lookup_table() {
                        let table_index = LookupTables::enum_index(&table);
                        lc += stage5_gammas[2 + table_index];
                    }
                    *o4 = lc;
                }
            });

        vals.map(MultilinearPolynomial::from)
    }

    /// Compute Val polynomials using pre-computed Zolt-compatible flags.
    ///
    /// This is equivalent to `compute_val_polys` but uses pre-computed flags
    /// instead of calling Jolt's `Flags` trait, because Zolt computes flags
    /// differently than Jolt (e.g., LUI: Zolt doesn't set AddOperands,
    /// JAL: Zolt sets WriteLookupOutputToRD but not AddOperands, etc.)
    #[cfg(feature = "zolt-debug")]
    #[allow(clippy::too_many_arguments)]
    fn compute_val_polys_zolt(
        zolt_flags: &[ZoltBytecodeFlags],
        eq_r_register_4: &[F],
        eq_r_register_5: &[F],
        stage1_gammas: &[F],
        stage2_gammas: &[F],
        stage3_gammas: &[F],
        stage4_gammas: &[F],
        stage5_gammas: &[F],
    ) -> [MultilinearPolynomial<F>; N_STAGES] {
        let K = zolt_flags.len();

        let mut vals: [Vec<F>; N_STAGES] = array::from_fn(|_| unsafe_allocate_zero_vec(K));
        let [v0, v1, v2, v3, v4] = &mut vals;

        zolt_flags
            .par_iter()
            .zip(v0.par_iter_mut())
            .zip(v1.par_iter_mut())
            .zip(v2.par_iter_mut())
            .zip(v3.par_iter_mut())
            .zip(v4.par_iter_mut())
            .enumerate()
            .for_each(|(k, (((((flags, o0), o1), o2), o3), o4))| {
                let circuit_flags = &flags.circuit_flags;
                let instr_flags = &flags.instruction_flags;

                if k < 20 || (k >= 30 && k <= 50) {
                    let imm_field_dbg = ZoltBytecodeFlags::encode_imm_field::<F>(flags);
                    use ark_serialize::CanonicalSerialize;
                    let mut imm_bytes = [0u8; 32];
                    imm_field_dbg.serialize_compressed(&mut imm_bytes[..]).ok();
                    let imm_hex: String = imm_bytes.iter().map(|b| format!("{:02x}", b)).collect();
                    eprintln!(
                        "[ZOLT_VAL_POLY] k={} addr=0x{:08x} opcode=0x{:02x} funct3={} rd={:?} rs1={:?} rs2={:?} imm={} imm_field_LE=[{}] cf={:?} if={:?} lt={:?} interleaved={}",
                        k, flags.address, flags.opcode, flags.funct3,
                        flags.rd, flags.rs1, flags.rs2,
                        flags.imm, imm_hex,
                        &circuit_flags, &instr_flags, flags.lookup_table_index,
                        flags.is_interleaved
                    );
                }

                // Stage 1 (Spartan outer sumcheck)
                // Val(k) = unexpanded_pc(k) + γ·imm(k) + γ²·cf[0](k) + ...
                // Per-opcode immediate encoding matching Zolt's stage6_prover.zig:
                //   ADDI(funct3=0)/ADDIW(funct3=0)/JAL/JALR → unsigned u64 bitcast
                //   LUI/AUIPC → truncated u32
                //   Everything else → signed field value (from_i128)
                {
                    let imm_field = ZoltBytecodeFlags::encode_imm_field::<F>(flags);
                    let mut lc = F::from_u64(flags.address as u64);
                    lc += imm_field * stage1_gammas[1];
                    for (i, flag) in circuit_flags.iter().enumerate() {
                        if *flag {
                            lc += stage1_gammas[2 + i];
                        }
                    }
                    *o0 = lc;
                }

                // Stage 2 (product virtualization)
                {
                    let mut lc = F::zero();
                    if circuit_flags[CircuitFlags::Jump as usize] {
                        lc += stage2_gammas[0];
                    }
                    if instr_flags[InstructionFlags::Branch as usize] {
                        lc += stage2_gammas[1];
                    }
                    if instr_flags[InstructionFlags::IsRdNotZero as usize] {
                        lc += stage2_gammas[2];
                    }
                    if circuit_flags[CircuitFlags::WriteLookupOutputToRD as usize] {
                        lc += stage2_gammas[3];
                    }
                    *o1 = lc;
                }

                // Stage 3 (Shift sumcheck)
                {
                    let imm_field = ZoltBytecodeFlags::encode_imm_field::<F>(flags);
                    let mut lc = imm_field;
                    lc += stage3_gammas[1].mul_u64(flags.address as u64);
                    if instr_flags[InstructionFlags::LeftOperandIsRs1Value as usize] {
                        lc += stage3_gammas[2];
                    }
                    if instr_flags[InstructionFlags::LeftOperandIsPC as usize] {
                        lc += stage3_gammas[3];
                    }
                    if instr_flags[InstructionFlags::RightOperandIsRs2Value as usize] {
                        lc += stage3_gammas[4];
                    }
                    if instr_flags[InstructionFlags::RightOperandIsImm as usize] {
                        lc += stage3_gammas[5];
                    }
                    if instr_flags[InstructionFlags::IsNoop as usize] {
                        lc += stage3_gammas[6];
                    }
                    if circuit_flags[CircuitFlags::VirtualInstruction as usize] {
                        lc += stage3_gammas[7];
                    }
                    if circuit_flags[CircuitFlags::IsFirstInSequence as usize] {
                        lc += stage3_gammas[8];
                    }
                    *o2 = lc;
                }

                // Stage 4 (registers read/write checking)
                // Use Zolt's rd/rs1/rs2 values (raw instruction bits, NOT normalized)
                {
                    let rd_eq = flags.rd
                        .filter(|&r| (r as usize) < eq_r_register_4.len())
                        .map_or(F::zero(), |r| eq_r_register_4[r as usize]);
                    let rs1_eq = flags.rs1
                        .filter(|&r| (r as usize) < eq_r_register_4.len())
                        .map_or(F::zero(), |r| eq_r_register_4[r as usize]);
                    let rs2_eq = flags.rs2
                        .filter(|&r| (r as usize) < eq_r_register_4.len())
                        .map_or(F::zero(), |r| eq_r_register_4[r as usize]);
                    *o3 = rd_eq * stage4_gammas[0]
                        + rs1_eq * stage4_gammas[1]
                        + rs2_eq * stage4_gammas[2];
                }

                // Stage 5 (registers val-evaluation + instruction lookups)
                // Use Zolt's rd value (raw instruction bits)
                {
                    let mut lc = flags.rd
                        .filter(|&r| (r as usize) < eq_r_register_5.len())
                        .map_or(F::zero(), |r| eq_r_register_5[r as usize]);
                    if !flags.is_interleaved {
                        lc += stage5_gammas[1];
                    }
                    if let Some(table_index) = flags.lookup_table_index {
                        lc += stage5_gammas[2 + table_index as usize];
                    }
                    *o4 = lc;
                }
            });

        // Debug: print all vals for each stage in hex LE format
        for s in 0..5 {
            for k in 0..K {
                use ark_serialize::CanonicalSerialize;
                let mut bytes = [0u8; 32];
                vals[s][k].serialize_compressed(&mut bytes[..]).ok();
                let hex_str: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
                eprintln!("[JOLT_VAL_POLY] Val[{}][{}]_LE=[{}]", s, k, hex_str);
            }
        }

        vals.map(MultilinearPolynomial::from)
    }

    fn compute_rv_claim_1(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        gamma_powers: &[F],
    ) -> F {
        let (_, unexpanded_pc_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, imm_claim) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);

        let circuit_flag_claims: Vec<F> = CircuitFlags::iter()
            .map(|flag| {
                opening_accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::OpFlags(flag),
                        SumcheckId::SpartanOuter,
                    )
                    .1
            })
            .collect();

        std::iter::once(unexpanded_pc_claim)
            .chain(std::iter::once(imm_claim))
            .chain(circuit_flag_claims)
            .zip_eq(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }

    fn compute_rv_claim_2(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        gamma_powers: &[F],
    ) -> F {
        let (_, jump_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::Jump),
            SumcheckId::SpartanProductVirtualization,
        );
        let (_, branch_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
            SumcheckId::SpartanProductVirtualization,
        );
        let (_, rd_wa_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsRdNotZero),
            SumcheckId::SpartanProductVirtualization,
        );
        let (_, write_lookup_output_to_rd_flag_claim) = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
                SumcheckId::SpartanProductVirtualization,
            );

        [
            jump_claim,
            branch_claim,
            rd_wa_claim,
            write_lookup_output_to_rd_flag_claim,
        ]
        .into_iter()
        .zip_eq(gamma_powers)
        .map(|(claim, gamma)| claim * gamma)
        .sum()
    }

    fn compute_rv_claim_3(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        gamma_powers: &[F],
    ) -> F {
        let (_, imm_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Imm,
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, spartan_shift_unexpanded_pc_claim) = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::UnexpandedPC,
                SumcheckId::SpartanShift,
            );
        let (_, instruction_input_unexpanded_pc_claim) = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::UnexpandedPC,
                SumcheckId::InstructionInputVirtualization,
            );

        assert_eq!(
            spartan_shift_unexpanded_pc_claim,
            instruction_input_unexpanded_pc_claim
        );

        let unexpanded_pc_claim = spartan_shift_unexpanded_pc_claim;
        let (_, left_is_rs1_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, left_is_pc_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsPC),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, right_is_rs2_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsRs2Value),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, right_is_imm_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, is_noop_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
        );
        let (_, is_virtual_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
        );
        let (_, is_first_in_sequence_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
        );

        [
            imm_claim,
            unexpanded_pc_claim,
            left_is_rs1_claim,
            left_is_pc_claim,
            right_is_rs2_claim,
            right_is_imm_claim,
            is_noop_claim,
            is_virtual_claim,
            is_first_in_sequence_claim,
        ]
        .into_iter()
        .zip_eq(gamma_powers)
        .map(|(claim, gamma)| claim * gamma)
        .sum()
    }

    fn compute_rv_claim_4(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        gamma_powers: &[F],
    ) -> F {
        std::iter::empty()
            .chain(once(VirtualPolynomial::RdWa))
            .chain(once(VirtualPolynomial::Rs1Ra))
            .chain(once(VirtualPolynomial::Rs2Ra))
            .map(|vp| {
                opening_accumulator
                    .get_virtual_polynomial_opening(vp, SumcheckId::RegistersReadWriteChecking)
                    .1
            })
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum::<F>()
    }

    fn compute_rv_claim_5(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        gamma_powers: &[F],
    ) -> F {
        let (_, rd_wa_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );

        let (_, raf_flag_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
        );

        let mut sum = rd_wa_claim * gamma_powers[0];
        sum += raf_flag_claim * gamma_powers[1];

        // Add lookup table flag claims from InstructionReadRaf
        for i in 0..LookupTables::<XLEN>::COUNT {
            let (_, claim) = opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
            );
            sum += claim * gamma_powers[2 + i];
        }

        sum
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for BytecodeReadRafSumcheckParams<F> {
    fn degree(&self) -> usize {
        self.d + 1
    }

    fn num_rounds(&self) -> usize {
        self.log_K + self.log_T
    }

    fn input_claim(&self, _: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn normalize_opening_point(
        &self,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut r = sumcheck_challenges.to_vec();
        r[0..self.log_K].reverse();
        r[self.log_K..].reverse();
        OpeningPoint::new(r)
    }
}
