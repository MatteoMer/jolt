use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_bitwise(target_dir);

    let shared_preprocessing = guest::preprocess_shared_bitwise(&mut program);
    let prover_preprocessing = guest::preprocess_prover_bitwise(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_bitwise(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
    );

    let prove = guest::build_prover_bitwise(program, prover_preprocessing);
    let verify = guest::build_verifier_bitwise(verifier_preprocessing);

    let a: u32 = 0xF0F0F0F0;
    let b: u32 = 0x0F0F0F0F;

    let now = Instant::now();
    let (output, proof, program_io) = prove(a, b);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    let is_valid = verify(a, b, output, program_io.panic, proof);

    info!("output: {output}");
    info!("valid: {is_valid}");
}
