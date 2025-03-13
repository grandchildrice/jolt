use std::time::Instant;

pub fn main() {
    // Creating test data
    let test_data: [[u8; 2]; 4] = [
        [20, 10], // feature1 = 20 (< T1 = 50) && feature2 = 10 (< T2 = 30) => V1 (10)
        [20, 40], // feature1 = 20 (< T1 = 50) && feature2 = 40 (>= T2 = 30) => V2 (20)
        [70, 60], // feature1 = 70 (>= T1 = 50) && feature2 = 60 (< T3 = 70) => V3 (30)
        [70, 80], // feature1 = 70 (>= T1 = 50) && feature2 = 80 (>= T3 = 70) => V4 (40)
    ];

    let (prove_predict, verify_predict) = guest::build_predict();
    let program_summary = guest::analyze_predict(test_data.clone());
    program_summary
        .write_to_file("gbm.txt".into())
        .expect("should write");

    let now = Instant::now();
    let (output, proof) = prove_predict(test_data.clone());
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_predict(proof);

    println!("output: {}", output);
    println!("valid: {}", is_valid);
}
