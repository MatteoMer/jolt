use jolt_sdk::serialize_and_print_size;
use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();

    // Check for --verify-zolt-proof option
    let zolt_proof_path = args.iter().position(|arg| arg == "--verify-zolt-proof")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    // Check for --zolt-preprocessing option
    let zolt_preprocessing_path = args.iter().position(|arg| arg == "--zolt-preprocessing")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    if let Some(proof_path) = zolt_proof_path {
        verify_zolt_proof(proof_path, zolt_preprocessing_path);
        return;
    }

    let save_to_disk = args.iter().any(|arg| arg == "--save");

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_fib(target_dir);

    let shared_preprocessing = guest::preprocess_shared_fib(&mut program);

    let prover_preprocessing = guest::preprocess_prover_fib(shared_preprocessing.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        guest::preprocess_verifier_fib(shared_preprocessing, verifier_setup);

    if save_to_disk {
        serialize_and_print_size(
            "Verifier Preprocessing",
            "/tmp/jolt_verifier_preprocessing.dat",
            &verifier_preprocessing,
        )
        .expect("Could not serialize preprocessing.");
    }

    let prove_fib = guest::build_prover_fib(program, prover_preprocessing);
    let verify_fib = guest::build_verifier_fib(verifier_preprocessing);

    let program_summary = guest::analyze_fib(10);
    program_summary
        .write_to_file("fib_10.txt".into())
        .expect("should write");

    let trace_file = "/tmp/fib_trace.bin";
    guest::trace_fib_to_file(trace_file, 50);
    info!("Trace file written to: {trace_file}.");

    let now = Instant::now();
    let (output, proof, io_device) = prove_fib(50);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    if save_to_disk {
        serialize_and_print_size("Proof", "/tmp/fib_proof.bin", &proof)
            .expect("Could not serialize proof.");
        serialize_and_print_size("io_device", "/tmp/fib_io_device.bin", &io_device)
            .expect("Could not serialize io_device.");
    }

    let is_valid = verify_fib(50, output, io_device.panic, proof);
    info!("output: {output}");
    info!("valid: {is_valid}");
}

fn verify_zolt_proof(proof_path: &str, zolt_preprocessing_path: Option<&str>) {
    use jolt_sdk::JoltDevice;
    use std::fs::File;
    use std::io::Read;

    println!("Verifying Zolt proof from: {}", proof_path);

    // First, build Jolt's standard preprocessing (we always need this for the commitment setup)
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_fib(target_dir);
    let mut shared_preprocessing = guest::preprocess_shared_fib(&mut program);

    // If --zolt-preprocessing is provided, override the RAM preprocessing
    if let Some(pp_path) = zolt_preprocessing_path {
        println!("Loading Zolt preprocessing from: {}", pp_path);

        let mut pp_file = File::open(pp_path).expect("Failed to open preprocessing file");
        let mut pp_bytes = Vec::new();
        pp_file.read_to_end(&mut pp_bytes).expect("Failed to read preprocessing file");
        println!("Read {} preprocessing bytes", pp_bytes.len());

        // Parse Zolt's preprocessing binary format
        use jolt_core::zkvm::ram::RAMPreprocessing;
        let mut cursor = std::io::Cursor::new(&pp_bytes);
        let mut buf8 = [0u8; 8];

        // Read min_bytecode_address
        cursor.read_exact(&mut buf8).expect("read min_bytecode_address");
        let min_bytecode_address = u64::from_le_bytes(buf8);
        println!("  min_bytecode_address: 0x{:016x}", min_bytecode_address);

        // Read bytecode_words
        cursor.read_exact(&mut buf8).expect("read bytecode_words len");
        let bytecode_words_len = u64::from_le_bytes(buf8) as usize;
        println!("  bytecode_words.len(): {}", bytecode_words_len);

        let mut bytecode_words = Vec::with_capacity(bytecode_words_len);
        for _ in 0..bytecode_words_len {
            cursor.read_exact(&mut buf8).expect("read bytecode word");
            bytecode_words.push(u64::from_le_bytes(buf8));
        }
        if bytecode_words_len > 0 {
            println!("  bytecode_words first 3: {:?}", &bytecode_words[..std::cmp::min(3, bytecode_words_len)]);
        }

        // Read memory_layout from Zolt preprocessing (20 u64 fields in order)
        let read_u64 = |c: &mut std::io::Cursor<&Vec<u8>>| -> u64 {
            let mut b = [0u8; 8];
            c.read_exact(&mut b).expect("read u64");
            u64::from_le_bytes(b)
        };

        let program_size = read_u64(&mut cursor);
        let max_trusted_advice_size = read_u64(&mut cursor);
        let trusted_advice_start = read_u64(&mut cursor);
        let trusted_advice_end = read_u64(&mut cursor);
        let max_untrusted_advice_size = read_u64(&mut cursor);
        let untrusted_advice_start = read_u64(&mut cursor);
        let untrusted_advice_end = read_u64(&mut cursor);
        let max_input_size = read_u64(&mut cursor);
        let max_output_size = read_u64(&mut cursor);
        let input_start = read_u64(&mut cursor);
        let input_end = read_u64(&mut cursor);
        let output_start = read_u64(&mut cursor);
        let output_end = read_u64(&mut cursor);
        let stack_size = read_u64(&mut cursor);
        let stack_end = read_u64(&mut cursor);
        let memory_size = read_u64(&mut cursor);
        let memory_end = read_u64(&mut cursor);
        let panic_address = read_u64(&mut cursor);
        let termination_address = read_u64(&mut cursor);
        let io_end = read_u64(&mut cursor);

        println!("  Zolt memory layout:");
        println!("    program_size: 0x{:x}", program_size);
        println!("    input_start: 0x{:016x}", input_start);
        println!("    output_start: 0x{:016x}", output_start);
        println!("    panic_address: 0x{:016x}", panic_address);
        println!("    termination_address: 0x{:016x}", termination_address);
        println!("    io_end: 0x{:016x}", io_end);

        // Override RAM preprocessing with Zolt's
        shared_preprocessing.ram = RAMPreprocessing {
            min_bytecode_address,
            bytecode_words,
        };

        // Override memory layout with Zolt's layout (all fields)
        shared_preprocessing.memory_layout.program_size = program_size;
        shared_preprocessing.memory_layout.max_trusted_advice_size = max_trusted_advice_size;
        shared_preprocessing.memory_layout.trusted_advice_start = trusted_advice_start;
        shared_preprocessing.memory_layout.trusted_advice_end = trusted_advice_end;
        shared_preprocessing.memory_layout.max_untrusted_advice_size = max_untrusted_advice_size;
        shared_preprocessing.memory_layout.untrusted_advice_start = untrusted_advice_start;
        shared_preprocessing.memory_layout.untrusted_advice_end = untrusted_advice_end;
        shared_preprocessing.memory_layout.max_input_size = max_input_size;
        shared_preprocessing.memory_layout.max_output_size = max_output_size;
        shared_preprocessing.memory_layout.input_start = input_start;
        shared_preprocessing.memory_layout.input_end = input_end;
        shared_preprocessing.memory_layout.output_start = output_start;
        shared_preprocessing.memory_layout.output_end = output_end;
        shared_preprocessing.memory_layout.stack_size = stack_size;
        shared_preprocessing.memory_layout.stack_end = stack_end;
        shared_preprocessing.memory_layout.memory_size = memory_size;
        shared_preprocessing.memory_layout.memory_end = memory_end;
        shared_preprocessing.memory_layout.panic = panic_address;
        shared_preprocessing.memory_layout.termination = termination_address;
        shared_preprocessing.memory_layout.io_end = io_end;

        // Read bytecode code_size (u64) - appended after memory_layout
        // This is the bytecode_K that must match the proof
        let zolt_code_size = if cursor.position() < pp_bytes.len() as u64 {
            let cs = read_u64(&mut cursor) as usize;
            println!("  Zolt bytecode code_size: {}", cs);
            Some(cs)
        } else {
            println!("  No bytecode code_size in preprocessing (old format)");
            None
        };

        // Read raw ELF program bytes for bytecode reconstruction
        let zolt_bytecode = if cursor.position() < pp_bytes.len() as u64 {
            let elf_base_address = read_u64(&mut cursor);
            let elf_program_len = read_u64(&mut cursor) as usize;
            let pos = cursor.position() as usize;
            if pos + elf_program_len <= pp_bytes.len() {
                let raw_bytes = &pp_bytes[pos..pos + elf_program_len];
                cursor.set_position((pos + elf_program_len) as u64);
                println!("  Zolt ELF: base=0x{:x} len={}", elf_base_address, elf_program_len);

                // Decode RISC-V instructions from raw ELF bytes
                use tracer::instruction::Instruction as TracerInstruction;
                let mut instructions = vec![TracerInstruction::NoOp]; // Prepend NoOp at index 0
                let mut raw_words: Vec<u32> = vec![0]; // NoOp has no raw word
                let mut offset: usize = 0;
                while offset < raw_bytes.len() {
                    let addr = elf_base_address + offset as u64;
                    if offset + 2 > raw_bytes.len() { break; }
                    let lo16 = u16::from_le_bytes([raw_bytes[offset], raw_bytes[offset + 1]]);
                    let is_compressed = (lo16 & 0x03) != 0x03;
                    if is_compressed {
                        // 16-bit compressed instruction: expand to 32-bit form
                        // For now, decode as a 32-bit instruction with only lower 16 bits
                        let word32 = lo16 as u32;
                        match TracerInstruction::decode(word32, addr, true) {
                            Ok(instr) => {
                                instructions.push(instr);
                                raw_words.push(word32);
                            }
                            Err(_) => {
                                instructions.push(TracerInstruction::NoOp);
                                raw_words.push(0);
                            }
                        }
                        offset += 2;
                    } else {
                        if offset + 4 > raw_bytes.len() { break; }
                        let word32 = u32::from_le_bytes([
                            raw_bytes[offset], raw_bytes[offset + 1],
                            raw_bytes[offset + 2], raw_bytes[offset + 3],
                        ]);
                        match TracerInstruction::decode(word32, addr, false) {
                            Ok(instr) => {
                                instructions.push(instr);
                                raw_words.push(word32);
                            }
                            Err(e) => {
                                println!("  WARNING: Failed to decode instruction at 0x{:x}: {} (word=0x{:08x})", addr, e, word32);
                                instructions.push(TracerInstruction::NoOp);
                                raw_words.push(0);
                            }
                        }
                        offset += 4;
                    }
                }
                println!("  Decoded {} raw instructions from ELF", instructions.len() - 1);

                // Expand virtual sequences (ADDIW → ADDI + VirtualSignExtendWord, etc.)
                // This matches what guest::decode does via inline_sequence.
                // The NoOp at index 0 returns empty from inline_sequence, so we
                // handle it separately: prepend NoOp, then expand the rest.
                use tracer::utils::virtual_registers::VirtualRegisterAllocator;
                use tracer::emulator::cpu::Xlen;
                let allocator = VirtualRegisterAllocator::default();
                let xlen = Xlen::Bit64; // Zolt targets RV64
                let mut expanded_instructions = vec![TracerInstruction::NoOp]; // index 0 = NoOp
                let mut expanded_raw_words: Vec<u32> = vec![0]; // NoOp has no raw word
                for (i, instr) in instructions.iter().enumerate().skip(1) {
                    // Skip the manually-prepended NoOp at index 0
                    let expanded = instr.inline_sequence(&allocator, xlen);
                    let parent_word = if i < raw_words.len() { raw_words[i] } else { 0 };
                    for exp_instr in expanded {
                        expanded_raw_words.push(parent_word);
                        expanded_instructions.push(exp_instr);
                    }
                }
                println!("  Expanded from {} raw to {} instructions (with virtual sequences)",
                    instructions.len() - 1, expanded_instructions.len() - 1);

                Some((expanded_instructions, expanded_raw_words))
            } else {
                println!("  WARNING: Not enough bytes for ELF program data");
                None
            }
        } else {
            println!("  No ELF program data in preprocessing (old format)");
            None
        };

        // Read termination_base_pc (u64) - bytecode index where LUI/ADDI/SB entries start
        let zolt_termination_base_pc = if cursor.position() < pp_bytes.len() as u64 {
            let tbpc = read_u64(&mut cursor) as usize;
            println!("  Zolt termination_base_pc: {}", tbpc);
            Some(tbpc)
        } else {
            println!("  No termination_base_pc in preprocessing (old format)");
            None
        };

        // Split zolt_bytecode into instructions and raw words
        let (zolt_instructions, zolt_raw_words) = match zolt_bytecode {
            Some((instrs, words)) => (Some(instrs), Some(words)),
            None => (None, None),
        };

        // Override bytecode preprocessing code_size and instruction array to match Zolt's program
        if let Some(cs) = zolt_code_size {
            println!("  Overriding bytecode code_size: {} -> {}", shared_preprocessing.bytecode.code_size, cs);
            use tracer::instruction::Instruction as TracerInstruction;
            let bytecode_mut = std::sync::Arc::make_mut(&mut shared_preprocessing.bytecode);
            bytecode_mut.code_size = cs;

            // If we have Zolt's decoded bytecode, use it instead of Jolt's
            if let Some(ref zolt_bc) = zolt_instructions {
                println!("  Replacing Jolt bytecode with decoded Zolt ELF bytecode ({} instrs)", zolt_bc.len());
                bytecode_mut.bytecode = zolt_bc.clone();
                // Rebuild PC map BEFORE padding (matching BytecodePreprocessing::preprocess behavior)
                bytecode_mut.pc_map = jolt_core::zkvm::bytecode::BytecodePCMapper::new(&bytecode_mut.bytecode);
            }

            // Resize/pad to match code_size (AFTER pc_map construction)
            bytecode_mut.bytecode.resize(cs, TracerInstruction::NoOp);

            // Store raw words in global static for Zolt Val polynomial computation
            if let Some(ref words) = zolt_raw_words {
                let mut padded_words = words.clone();
                padded_words.resize(cs, 0);
                let _ = jolt_core::zkvm::bytecode::ZOLT_RAW_WORDS.set(padded_words);
                println!("  Stored {} raw instruction words for Zolt flag computation", words.len());
            }

            // Store termination address for bytecode entry reconstruction
            let _ = jolt_core::zkvm::bytecode::ZOLT_TERMINATION_ADDRESS.set(termination_address);
            println!("  Stored termination_address=0x{:x} for termination bytecode entries", termination_address);

            // Store termination_base_pc for separate LUI/ADDI/SB entries
            if let Some(tbpc) = zolt_termination_base_pc {
                let _ = jolt_core::zkvm::bytecode::ZOLT_TERMINATION_BASE_PC.set(tbpc);
                println!("  Stored termination_base_pc={} for separate termination entries", tbpc);
            }
        }

        println!("  Overrode shared_preprocessing.ram, memory_layout, and bytecode with Zolt values");
        println!("  Final bytecode_words.len(): {}", shared_preprocessing.ram.bytecode_words.len());
    } else {
        println!("Using Jolt's own fibonacci preprocessing (no --zolt-preprocessing)");
    }

    println!("  Bytecode code_size: {}", shared_preprocessing.bytecode.code_size);
    println!("  Bytecode bytecode.len(): {}", shared_preprocessing.bytecode.bytecode.len());
    println!("  bytecode_K: {}", shared_preprocessing.bytecode.code_size);
    // Print first 20 bytecode entries for comparison with Zolt
    for (idx, instr) in shared_preprocessing.bytecode.bytecode.iter().enumerate().take(20) {
        let norm = instr.normalize();
        let rd = norm.operands.rd.unwrap_or(0);
        let rs1 = norm.operands.rs1.unwrap_or(0);
        let rs2 = norm.operands.rs2.unwrap_or(0);
        let imm = norm.operands.imm;
        let is_noop = matches!(instr, tracer::instruction::Instruction::NoOp);
        println!("  bytecode[{}]: addr=0x{:08x} rd={} rs1={} rs2={} imm={} noop={} instr={:?}",
            idx, norm.address, rd, rs1, rs2, imm, is_noop, instr);
    }

    let prover_preprocessing = guest::preprocess_prover_fib(shared_preprocessing.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        guest::preprocess_verifier_fib(shared_preprocessing, verifier_setup);

    // Load the proof bytes from the file
    let mut file = File::open(proof_path).expect("Failed to open proof file");
    let mut proof_bytes = Vec::new();
    file.read_to_end(&mut proof_bytes).expect("Failed to read proof file");

    println!("Read {} bytes from proof file", proof_bytes.len());

    // Deserialize proof
    use ark_serialize::CanonicalDeserialize;
    use jolt_core::zkvm::RV64IMACProof;

    let proof = match RV64IMACProof::deserialize_compressed(&*proof_bytes) {
        Ok(p) => p,
        Err(e) => {
            println!("Failed to deserialize proof: {:?}", e);
            return;
        }
    };

    println!("Proof deserialized successfully!");
    println!("  trace_length: {}", proof.trace_length);
    println!("  ram_K: {}", proof.ram_K);
    println!("  bytecode_K: {}", proof.bytecode_K);

    // Create a JoltDevice matching the preprocessing
    let program_io = JoltDevice {
        inputs: Vec::new(),
        trusted_advice: Vec::new(),
        untrusted_advice: Vec::new(),
        outputs: Vec::new(),
        panic: false,
        memory_layout: verifier_preprocessing.shared.memory_layout.clone(),
    };

    // Create verifier and verify
    use jolt_core::zkvm::RV64IMACVerifier;

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        match RV64IMACVerifier::new(&verifier_preprocessing, proof, program_io, None, None) {
            Ok(verifier) => verifier.verify(),
            Err(e) => Err(e.into()),
        }
    }));

    match result {
        Ok(Ok(())) => println!("\n✓ Verification PASSED!"),
        Ok(Err(e)) => println!("\n✗ Verification FAILED: {:?}", e),
        Err(e) => println!("\n✗ Verification PANICKED: {:?}", e),
    }
}
