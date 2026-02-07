use crate::zkvm::instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use crate::zkvm::lookup_table::LookupTables;
use tracer::instruction::{slli::SLLI, RISCVCycle};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for SLLI {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        // SLLI has no direct lookup table in Zolt - it uses interleaved operands
        None
    }
}

impl Flags for SLLI {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        // SLLI has WriteLookupOutputToRD but NO Add/Sub/Mul operand flags
        flags[CircuitFlags::WriteLookupOutputToRD] = true;
        flags[CircuitFlags::VirtualInstruction] = self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsFirstInSequence] = self.is_first_in_sequence;
        flags[CircuitFlags::IsCompressed] = self.is_compressed;
        flags
    }

    fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS] {
        let mut flags = [false; NUM_INSTRUCTION_FLAGS];
        flags[InstructionFlags::LeftOperandIsRs1Value] = true;
        flags[InstructionFlags::RightOperandIsImm] = true;
        flags[InstructionFlags::IsRdNotZero] = self.operands.rd != 0;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<SLLI> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
