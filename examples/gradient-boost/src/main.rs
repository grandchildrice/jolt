use std::time::Instant;

pub fn main() {
    // Creating test data
    let test_data: [u8; 2] = [70, 80];

    // Run standard implementation
    run_standard_implementation(&test_data);

    // Run GBDT implementation
    run_gbdt_infer_implementation(&test_data);
}

fn run_standard_implementation(test_data: &[u8; 2]) {
    println!("\n=== Standard Implementation ===");

    let (prove_predict, verify_predict) = guest::build_predict();
    let program_summary = guest::analyze_predict(test_data.clone());
    program_summary
        .write_to_file("gbm_standard.txt".into())
        .expect("should write");

    let now = Instant::now();
    let (output, proof) = prove_predict(test_data.clone());
    let prover_time = now.elapsed().as_secs_f64();
    println!("Prover runtime: {} s", prover_time);

    let is_valid = verify_predict(proof);
    println!("Output: {}", output);
    println!("Valid: {}", is_valid);
}

fn run_gbdt_infer_implementation(test_data: &[u8; 2]) {
    println!("\n=== GBDT Implementation ===");

    let (prove_predict, verify_predict) = guest::build_predict_with_gbdt();
    let program_summary = guest::analyze_predict_with_gbdt(test_data.clone());
    program_summary
        .write_to_file("gbm_gbdt_infer.txt".into())
        .expect("should write");

    let now = Instant::now();
    let (output, proof) = prove_predict(test_data.clone());
    let prover_time = now.elapsed().as_secs_f64();
    println!("Prover runtime: {} s", prover_time);

    let is_valid = verify_predict(proof);
    println!("Output: {}", output);
    println!("Valid: {}", is_valid);
}
