use crate::{
    field::JoltField, jolt::subtable::ltu::LtuSubtable,
    utils::instruction_utils::chunk_and_concatenate_operands,
};
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

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        // SLTInstruction と同様の方法でサブテーブルの値を取得
        let vals_by_subtable = self.slice_values(vals, C, M);

        // 各サブテーブルの比較結果を取得
        let left_lt_t1 = vals_by_subtable[0][0]; // 左側の値 < T1
        let right_lt_t2 = vals_by_subtable[1][0]; // 右側の値 < T2
        let right_lt_t3 = vals_by_subtable[2][0]; // 右側の値 < T3

        // 決定木のロジックを多項式で表現
        // 各条件分岐に対する出力値を計算
        left_lt_t1 * right_lt_t2 * F::from_u64(V1 as u64) +  // 条件: left < T1 && right < T2
        left_lt_t1 * (F::one() - right_lt_t2) * F::from_u64(V2 as u64) +  // 条件: left < T1 && right >= T2
        (F::one() - left_lt_t1) * right_lt_t3 * F::from_u64(V3 as u64) +  // 条件: left >= T1 && right < T3
        (F::one() - left_lt_t1) * (F::one() - right_lt_t3) * F::from_u64(V4 as u64)
        // 条件: left >= T1 && right >= T3
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        3
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (Box::new(LtuSubtable::new()), SubtableIndices::from(0)),
            (Box::new(LtuSubtable::new()), SubtableIndices::from(0)),
            (Box::new(LtuSubtable::new()), SubtableIndices::from(0)),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        // 入力値を8ビットに制限
        let left = (self.0 & 0xFF) as u8;
        let right = (self.1 & 0xFF) as u8;

        // 各比較用にインデックスを生成
        let left_lt_t1_idx = (left as u16) | ((T1 as u16) << 8);
        let right_lt_t2_idx = (right as u16) | ((T2 as u16) << 8);
        let right_lt_t3_idx = (right as u16) | ((T3 as u16) << 8);

        // インデックスの配列を生成
        // chunk_and_concatenate_operands ではなく、直接インデックスを作成
        let mut indices = vec![
            left_lt_t1_idx as usize,
            right_lt_t2_idx as usize,
            right_lt_t3_idx as usize,
        ];

        // もしCが3より大きければ、残りを0で埋める
        indices.resize(C, 0);
        indices
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
    fn gradient_boost_e2e() {
        let mut rng = test_rng();
        const C: usize = 3;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let x = rng.next_u64() & 0xFF; // 8ビットに制限
            let y = rng.next_u64() & 0xFF; // 8ビットに制限
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
