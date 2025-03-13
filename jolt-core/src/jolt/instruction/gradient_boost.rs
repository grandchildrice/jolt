use crate::field::JoltField;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::JoltInstruction;
use crate::jolt::{
    instruction::SubtableIndices,
    subtable::{gradient_boost::GradientBoostSubtable, LassoSubtable},
};

// Threshold constants for comparison - must match subtable
const T1: u8 = 50; // Threshold for splitting feature1
const T2: u8 = 30; // Next threshold when feature1 < T1
const T3: u8 = 70; // Next threshold when feature1 >= T1

// Leaf output values - must match subtable
const V1: u8 = 10; // Output value 1
const V2: u8 = 20; // Output value 2
const V3: u8 = 30; // Output value 3
const V4: u8 = 40; // Output value 4

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct GradientBoostInstruction(pub u64, pub u64);

impl GradientBoostInstruction {
    // Decision tree inference function - identical to the one in subtable
    fn inference(left: u8, right: u8) -> u8 {
        if left < T1 {
            if right < T2 {
                V1
            } else {
                V2
            }
        } else {
            if right < T3 {
                V3
            } else {
                V4
            }
        }
    }
}

impl JoltInstruction for GradientBoostInstruction {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, _M: usize) -> F {
        let vals_by_subtable = self.slice_values(vals, C, 0);
        let predictions = vals_by_subtable[0];
        predictions[0]
    }

    fn g_poly_degree(&self, _C: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        _C: usize,
        _M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![(
            Box::new(GradientBoostSubtable::new()),
            SubtableIndices::from(0),
        )]
    }

    fn to_indices(&self, _C: usize, _log_M: usize) -> Vec<usize> {
        let left = (self.0 & 0xFF) as u8;
        let right = (self.1 & 0xFF) as u8;
        let idx = right as usize | ((left as usize) << 8);
        vec![idx]
    }

    fn lookup_entry(&self) -> u64 {
        let left = (self.0 & 0xFF) as u8;
        let right = (self.1 & 0xFF) as u8;
        Self::inference(left, right) as u64
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self(rng.next_u64() & 0xFF, rng.next_u64() & 0xFF)
    }
}

#[cfg(test)]
mod test {
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use super::GradientBoostInstruction;

    fn test_inference(left: u8, right: u8, expected: u8) {
        let instruction = GradientBoostInstruction(left as u64, right as u64);
        let result = instruction.lookup_entry();
        assert_eq!(
            result, expected as u64,
            "Inference failed for ({}, {})",
            left, right
        );
    }

    #[test]
    fn test_inference_logic() {
        test_inference(0, 0, 10); // left < T1, right < T2 => V1
        test_inference(10, 40, 20); // left < T1, right >= T2 => V2
        test_inference(60, 20, 30); // left >= T1, right < T3 => V3
        test_inference(90, 90, 40); // left >= T1, right >= T3 => V4

        // Test boundary conditions
        test_inference(49, 29, 10); // Borderline (V1)
        test_inference(49, 30, 20); // Borderline (V2)
        test_inference(50, 69, 30); // Borderline (V3)
        test_inference(50, 70, 40); // Borderline (V4)
    }

    // Test the basic implementation directly
    #[test]
    fn gradient_boost_d_e2e() {
        const C: usize = 1; // Changed to 1 since we only need one index
        const M: usize = 1 << 16;

        // Test the decision tree logic directly
        let instructions = vec![
            GradientBoostInstruction(0, 0),
            GradientBoostInstruction(10, 10),
            GradientBoostInstruction(49, 20),
            GradientBoostInstruction(40, 5),
            GradientBoostInstruction(10, 40),
            GradientBoostInstruction(25, 60),
            GradientBoostInstruction(45, 35),
            GradientBoostInstruction(60, 30),
            GradientBoostInstruction(100, 50),
            GradientBoostInstruction(80, 60),
            GradientBoostInstruction(100, 100),
            GradientBoostInstruction(75, 80),
            GradientBoostInstruction(90, 90),
        ];

        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    // Test the basic implementation directly
    #[test]
    fn gradient_boost_e2e() {
        let mut rng = test_rng();
        const C: usize = 1; // Changed to 1 since we only need one index
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = GradientBoostInstruction(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let instructions = vec![
            GradientBoostInstruction(0, 0),
            GradientBoostInstruction(10, 10),
            GradientBoostInstruction(49, 20),
            GradientBoostInstruction(40, 5),
            GradientBoostInstruction(10, 40),
            GradientBoostInstruction(25, 60),
            GradientBoostInstruction(45, 35),
            GradientBoostInstruction(60, 30),
            GradientBoostInstruction(100, 50),
            GradientBoostInstruction(80, 60),
            GradientBoostInstruction(100, 100),
            GradientBoostInstruction(75, 80),
            GradientBoostInstruction(90, 90),
        ];

        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
