use common::constants::virtual_register_index;
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use super::VirtualInstructionSequence;
use crate::jolt::instruction::{
    // add::ADDInstruction, beq::BEQInstruction, mul::MULInstruction,
    virtual_advice::ADVICEInstruction,
    JoltInstruction,
};

pub struct GradientBoostInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> GradientBoostInstruction<WORD_SIZE> {
    // Decision tree inference function - identical to the one in subtable
    fn inference(left: u64, right: u64) -> u64 {
        // Threshold constants for comparison - must match subtable
        const T1: u64 = 50; // Threshold for splitting feature1
        const T2: u64 = 30; // Next threshold when feature1 < T1
        const T3: u64 = 70; // Next threshold when feature1 >= T1

        // Leaf output values - must match subtable
        const V1: u64 = 10; // Output value 1
        const V2: u64 = 20; // Output value 2
        const V3: u64 = 30; // Output value 3
        const V4: u64 = 40; // Output value 4
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

impl<const WORD_SIZE: usize> VirtualInstructionSequence for GradientBoostInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 2;

    fn virtual_trace(trace_row: RVTraceRow) -> Vec<RVTraceRow> {
        // GBDT_INFR source registers
        // let r_x = trace_row.instruction.rs1;
        // let r_y = trace_row.instruction.rs2;
        // Virtual registers used in sequence
        let v_i = Some(virtual_register_index(0));

        let x = trace_row.register_state.rs1_val.unwrap();
        let y = trace_row.register_state.rs2_val.unwrap();

        let mut virtual_trace: Vec<RVTraceRow> = vec![];

        let inference = Self::inference(x, y);

        let i = ADVICEInstruction::<WORD_SIZE>(inference).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_i,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(i),
            },
            memory_state: None,
            advice_value: Some(inference),
            precompile_input: None,
            precompile_output_address: None,
        });

        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_MOVE,
                rs1: v_i,
                rs2: None,
                rd: trace_row.instruction.rd,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(i),
                rs2_val: None,
                rd_post_val: Some(i),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        virtual_trace
    }

    fn sequence_output(x: u64, y: u64) -> u64 {
        Self::inference(x, y)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{jolt::instruction::JoltInstruction, jolt_virtual_sequence_test};

    #[test]
    fn gradient_boost_sequence_32() {
        jolt_virtual_sequence_test!(GradientBoostInstruction<32>, RV32IM::GBDT_INFER);
    }
}
