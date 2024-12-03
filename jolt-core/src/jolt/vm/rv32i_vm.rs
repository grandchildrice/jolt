use crate::field::JoltField;
use crate::jolt::instruction::virtual_assert_aligned_memory_access::AssertAlignedMemoryAccessInstruction;
use crate::jolt::instruction::virtual_assert_valid_div0::AssertValidDiv0Instruction;
use crate::jolt::instruction::virtual_assert_valid_unsigned_remainder::AssertValidUnsignedRemainderInstruction;
use crate::jolt::instruction::virtual_move::MOVEInstruction;
use crate::jolt::subtable::div_by_zero::DivByZeroSubtable;
use crate::jolt::subtable::low_bit::LowBitSubtable;
use crate::jolt::subtable::right_is_zero::RightIsZeroSubtable;
use crate::poly::commitment::hyperkzg::HyperKZG;
use crate::r1cs::constraints::JoltRV32IMConstraints;
use crate::r1cs::inputs::JoltR1CSInputs;
use ark_bn254::{Bn254, Fr};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use enum_dispatch::enum_dispatch;
use rand::{prelude::StdRng, RngCore};
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use std::fs::File;
use std::io::{self, Write};

use super::{Jolt, JoltCommitments, JoltProof};
use crate::jolt::instruction::{
    add::ADDInstruction, and::ANDInstruction, beq::BEQInstruction, bge::BGEInstruction,
    bgeu::BGEUInstruction, bne::BNEInstruction, mul::MULInstruction, mulhu::MULHUInstruction,
    mulu::MULUInstruction, or::ORInstruction, sll::SLLInstruction, slt::SLTInstruction,
    sltu::SLTUInstruction, sra::SRAInstruction, srl::SRLInstruction, sub::SUBInstruction,
    virtual_advice::ADVICEInstruction, virtual_assert_lte::ASSERTLTEInstruction,
    virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction,
    virtual_movsign::MOVSIGNInstruction, xor::XORInstruction, JoltInstruction, JoltInstructionSet,
    SubtableIndices,
};
use crate::jolt::subtable::{
    and::AndSubtable, eq::EqSubtable, eq_abs::EqAbsSubtable, identity::IdentitySubtable,
    left_is_zero::LeftIsZeroSubtable, left_msb::LeftMSBSubtable, lt_abs::LtAbsSubtable,
    ltu::LtuSubtable, or::OrSubtable, right_msb::RightMSBSubtable, sign_extend::SignExtendSubtable,
    sll::SllSubtable, sra_sign::SraSignSubtable, srl::SrlSubtable,
    truncate_overflow::TruncateOverflowSubtable, xor::XorSubtable, JoltSubtableSet, LassoSubtable,
    SubtableId,
};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;

/// Generates an enum out of a list of JoltInstruction types. All JoltInstruction methods
/// are callable on the enum type via enum_dispatch.
macro_rules! instruction_set {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[allow(non_camel_case_types)]
        #[repr(u8)]
        #[derive(Copy, Clone, Debug, PartialEq, EnumIter, EnumCountMacro, Serialize, Deserialize)]
        #[enum_dispatch(JoltInstruction)]
        pub enum $enum_name {
            $($alias($struct)),+
        }
        impl JoltInstructionSet for $enum_name {}
        impl $enum_name {
            pub fn random_instruction(rng: &mut StdRng) -> Self {
                let index = rng.next_u64() as usize % $enum_name::COUNT;
                let instruction = $enum_name::iter()
                    .enumerate()
                    .filter(|(i, _)| *i == index)
                    .map(|(_, x)| x)
                    .next()
                    .unwrap();
                instruction.random(rng)
            }
        }
        // Need a default so that we can derive EnumIter on `JoltR1CSInputs`
        impl Default for $enum_name {
            fn default() -> Self {
                $enum_name::iter().collect::<Vec<_>>()[0]
            }
        }
    };
}

/// Generates an enum out of a list of LassoSubtable types. All LassoSubtable methods
/// are callable on the enum type via enum_dispatch.
macro_rules! subtable_enum {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[allow(non_camel_case_types)]
        #[repr(u8)]
        #[enum_dispatch(LassoSubtable<F>)]
        #[derive(EnumCountMacro, EnumIter)]
        pub enum $enum_name<F: JoltField> { $($alias($struct)),+ }
        impl<F: JoltField> From<SubtableId> for $enum_name<F> {
          fn from(subtable_id: SubtableId) -> Self {
            $(
              if subtable_id == TypeId::of::<$struct>() {
                $enum_name::from(<$struct>::new())
              } else
            )+
            { panic!("Unexpected subtable id {:?}", subtable_id) } // TODO(moodlezoup): better error handling
          }
        }

        impl<F: JoltField> From<$enum_name<F>> for usize {
            fn from(subtable: $enum_name<F>) -> usize {
                // Discriminant: https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
                let byte = unsafe { *(&subtable as *const $enum_name<F> as *const u8) };
                byte as usize
            }
        }
        impl<F: JoltField> JoltSubtableSet<F> for $enum_name<F> {}
    };
}

const WORD_SIZE: usize = 32;

instruction_set!(
  RV32I,
  ADD: ADDInstruction<WORD_SIZE>,
  SUB: SUBInstruction<WORD_SIZE>,
  AND: ANDInstruction<WORD_SIZE>,
  OR: ORInstruction<WORD_SIZE>,
  XOR: XORInstruction<WORD_SIZE>,
  BEQ: BEQInstruction<WORD_SIZE>,
  BGE: BGEInstruction<WORD_SIZE>,
  BGEU: BGEUInstruction<WORD_SIZE>,
  BNE: BNEInstruction<WORD_SIZE>,
  SLT: SLTInstruction<WORD_SIZE>,
  SLTU: SLTUInstruction<WORD_SIZE>,
  SLL: SLLInstruction<WORD_SIZE>,
  SRA: SRAInstruction<WORD_SIZE>,
  SRL: SRLInstruction<WORD_SIZE>,
  MOVSIGN: MOVSIGNInstruction<WORD_SIZE>,
  MUL: MULInstruction<WORD_SIZE>,
  MULU: MULUInstruction<WORD_SIZE>,
  MULHU: MULHUInstruction<WORD_SIZE>,
  VIRTUAL_ADVICE: ADVICEInstruction<WORD_SIZE>,
  VIRTUAL_MOVE: MOVEInstruction<WORD_SIZE>,
  VIRTUAL_ASSERT_LTE: ASSERTLTEInstruction<WORD_SIZE>,
  VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER: AssertValidSignedRemainderInstruction<WORD_SIZE>,
  VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER: AssertValidUnsignedRemainderInstruction<WORD_SIZE>,
  VIRTUAL_ASSERT_VALID_DIV0: AssertValidDiv0Instruction<WORD_SIZE>,
  VIRTUAL_ASSERT_HALFWORD_ALIGNMENT: AssertAlignedMemoryAccessInstruction<WORD_SIZE, 2>,
  VIRTUAL_ASSERT_WORD_ALIGNMENT: AssertAlignedMemoryAccessInstruction<WORD_SIZE, 4>
);
subtable_enum!(
  RV32ISubtables,
  AND: AndSubtable<F>,
  EQ_ABS: EqAbsSubtable<F>,
  EQ: EqSubtable<F>,
  LEFT_MSB: LeftMSBSubtable<F>,
  RIGHT_MSB: RightMSBSubtable<F>,
  IDENTITY: IdentitySubtable<F>,
  LT_ABS: LtAbsSubtable<F>,
  LTU: LtuSubtable<F>,
  OR: OrSubtable<F>,
  SIGN_EXTEND_16: SignExtendSubtable<F, 16>,
  SLL0: SllSubtable<F, 0, WORD_SIZE>,
  SLL1: SllSubtable<F, 1, WORD_SIZE>,
  SLL2: SllSubtable<F, 2, WORD_SIZE>,
  SLL3: SllSubtable<F, 3, WORD_SIZE>,
  SRA_SIGN: SraSignSubtable<F, WORD_SIZE>,
  SRL0: SrlSubtable<F, 0, WORD_SIZE>,
  SRL1: SrlSubtable<F, 1, WORD_SIZE>,
  SRL2: SrlSubtable<F, 2, WORD_SIZE>,
  SRL3: SrlSubtable<F, 3, WORD_SIZE>,
  TRUNCATE: TruncateOverflowSubtable<F, WORD_SIZE>,
  XOR: XorSubtable<F>,
  LEFT_IS_ZERO: LeftIsZeroSubtable<F>,
  RIGHT_IS_ZERO: RightIsZeroSubtable<F>,
  DIV_BY_ZERO: DivByZeroSubtable<F>,
  LSB: LowBitSubtable<F, 0>,
  SECOND_LEAST_SIGNIFICANT_BIT: LowBitSubtable<F, 1>
);

// ==================== JOLT ====================

pub enum RV32IJoltVM {}

pub const C: usize = 4;
pub const M: usize = 1 << 16;

impl<F, PCS, ProofTranscript> Jolt<F, PCS, C, M, ProofTranscript> for RV32IJoltVM
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type InstructionSet = RV32I;
    type Subtables = RV32ISubtables<F>;
    type Constraints = JoltRV32IMConstraints;
}

pub type RV32IJoltProof<F, PCS, ProofTranscript> =
    JoltProof<C, M, JoltR1CSInputs, F, PCS, RV32I, RV32ISubtables<F>, ProofTranscript>;

use crate::utils::transcript::{KeccakTranscript, Transcript};
use eyre::Result;
use std::io::Cursor;
use std::path::PathBuf;

pub trait Serializable: CanonicalSerialize + CanonicalDeserialize + Sized {
    /// Gets the byte size of the serialized data
    fn size(&self) -> Result<usize> {
        let mut buffer = Vec::new();
        self.serialize_compressed(&mut buffer)?;
        Ok(buffer.len())
    }

    /// Saves the data to a file
    fn save_to_file<P: Into<PathBuf>>(&self, path: P) -> Result<()> {
        let file = File::create(path.into())?;
        self.serialize_compressed(file)?;
        Ok(())
    }

    /// Reads data from a file
    fn from_file<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let file = File::open(path.into())?;
        Ok(Self::deserialize_compressed(file)?)
    }

    /// Serializes the data to a byte vector
    fn serialize_to_bytes(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        self.serialize_compressed(&mut buffer)?;
        Ok(buffer)
    }

    /// Deserializes data from a byte vector
    fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self> {
        let cursor = Cursor::new(bytes);
        Ok(Self::deserialize_compressed(cursor)?)
    }
}

pub type ProofTranscript = KeccakTranscript;
pub type PCS = HyperKZG<Bn254, ProofTranscript>;
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltHyperKZGProof {
    pub proof: RV32IJoltProof<Fr, PCS, ProofTranscript>,
    pub commitments: JoltCommitments<PCS, ProofTranscript>,
}

impl Serializable for JoltHyperKZGProof {}

// ==================== TEST ====================

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr, G1Projective};

    use std::collections::HashSet;
    use std::fs::File;
    use std::io::Write;

    use crate::field::JoltField;
    use crate::host;
    use crate::jolt::instruction::JoltInstruction;
    use crate::jolt::vm::rv32i_vm::{Jolt, RV32IJoltVM, C, M};
    use crate::poly::commitment::commitment_scheme::CommitmentScheme;
    use crate::poly::commitment::hyperkzg::HyperKZG;
    use crate::poly::commitment::hyrax::HyraxScheme;
    use crate::poly::commitment::mock::MockCommitScheme;
    use crate::poly::commitment::zeromorph::Zeromorph;
    use crate::utils::transcript::{KeccakTranscript, Transcript};
    use std::sync::Mutex;
    use strum::{EnumCount, IntoEnumIterator};

    // If multiple tests try to read the same trace artifacts simultaneously, they will fail
    lazy_static::lazy_static! {
        static ref FIB_FILE_LOCK: Mutex<()> = Mutex::new(());
        static ref SHA3_FILE_LOCK: Mutex<()> = Mutex::new(());
    }

    fn test_instruction_set_subtables<PCS, ProofTranscript>()
    where
        PCS: CommitmentScheme<ProofTranscript>,
        ProofTranscript: Transcript,
    {
        let mut subtable_set: HashSet<_> = HashSet::new();
        for instruction in
            <RV32IJoltVM as Jolt<_, PCS, C, M, ProofTranscript>>::InstructionSet::iter()
        {
            for (subtable, _) in instruction.subtables::<Fr>(C, M) {
                // panics if subtable cannot be cast to enum variant
                let _ = <RV32IJoltVM as Jolt<_, PCS, C, M, ProofTranscript>>::Subtables::from(
                    subtable.subtable_id(),
                );
                subtable_set.insert(subtable.subtable_id());
            }
        }
        assert_eq!(
            subtable_set.len(),
            <RV32IJoltVM as Jolt<_, PCS, C, M, ProofTranscript>>::Subtables::COUNT,
            "Unused enum variants in Subtables"
        );
    }

    #[test]
    fn instruction_set_subtables() {
        test_instruction_set_subtables::<
            HyraxScheme<G1Projective, KeccakTranscript>,
            KeccakTranscript,
        >();
        test_instruction_set_subtables::<Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript>();
        test_instruction_set_subtables::<HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>();
    }

    fn fib_e2e<F, PCS, ProofTranscript>()
    where
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        let artifact_guard = FIB_FILE_LOCK.lock().unwrap();
        let mut program = host::Program::new("fibonacci-guest");
        program.set_input(&9u32);

        let segmentation_enable = true;

        if segmentation_enable {
            let (bytecode, memory_init) = program.decode();
            let (io_device, trace) = {
                let (io_device, segmentations) = program.segment_trace();
                let raw_register_init = segmentations[1].0.0.clone();

                // save register_init to a file
                // let encoded = bincode::serialize(&raw_register_init).expect("Failed to serialize");
                // let mut file = File::create("config.bin").expect("Failed to create");
                // file.write_all(&encoded).expect("Failed to write");

                (io_device, segmentations[1].1.clone())
            };
            drop(artifact_guard);
    
            let is_final_segment = false;
    
            let preprocessing = RV32IJoltVM::preprocess(
                bytecode.clone(),
                io_device.memory_layout.clone(),
                memory_init,
                1 << 20,
                1 << 20,
                1 << 20,
            );
            let (proof, commitments, debug_info) =
            <RV32IJoltVM as Jolt<F, PCS, C, M, ProofTranscript>>::segment_prove(
                io_device,
                trace,
                preprocessing.clone(),
                is_final_segment, // セグメントの最後か？ もし最後なら、program_ioのoutputのOutputSumcheckProofで一致を証明する。
            );
    
            let verification_result = if is_final_segment {
                RV32IJoltVM::verify(preprocessing, proof, commitments, debug_info)
            } else {
                RV32IJoltVM::segment_verify(preprocessing, proof, commitments, debug_info)
            };
            // let verification_result = RV32IJoltVM::verify(preprocessing, proof, commitments, debug_info);
            assert!(
                verification_result.is_ok(),
                "Verification failed with error: {:?}",
                verification_result.err()
            );
        } else {
            let (bytecode, memory_init) = program.decode();
            let (io_device, trace) = program.trace();
            drop(artifact_guard);
    
            let preprocessing = RV32IJoltVM::preprocess(
                bytecode.clone(),
                io_device.memory_layout.clone(),
                memory_init,
                1 << 20,
                1 << 20,
                1 << 20,
            );
            let (proof, commitments, debug_info) =
                <RV32IJoltVM as Jolt<F, PCS, C, M, ProofTranscript>>::prove(
                    io_device,
                    trace,
                    preprocessing.clone(),
                );
            let verification_result = RV32IJoltVM::verify(preprocessing, proof, commitments, debug_info);
            assert!(
                verification_result.is_ok(),
                "Verification failed with error: {:?}",
                verification_result.err()
            );
        }
    }

    #[test]
    fn fib_e2e_mock() {
        fib_e2e::<Fr, MockCommitScheme<Fr, KeccakTranscript>, KeccakTranscript>();
    }

    #[ignore = "Opening proof reduction for Hyrax doesn't work right now"]
    #[test]
    fn fib_e2e_hyrax() {
        fib_e2e::<
            ark_bn254::Fr,
            HyraxScheme<ark_bn254::G1Projective, KeccakTranscript>,
            KeccakTranscript,
        >();
    }

    #[test]
    fn fib_e2e_zeromorph() {
        fib_e2e::<Fr, Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript>();
    }

    #[test]
    fn fib_e2e_hyperkzg() {
        fib_e2e::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>();
    }

    // TODO(sragss): Finish Binius.
    // #[test]
    // fn fib_e2e_binius() {
    //     type Field = crate::field::binius::BiniusField<binius_field::BinaryField128b>;
    //     fib_e2e::<Field, MockCommitScheme<Field>>();
    // }

    #[ignore = "Opening proof reduction for Hyrax doesn't work right now"]
    #[test]
    fn muldiv_e2e_hyrax() {
        let mut program = host::Program::new("muldiv-guest");
        program.set_input(&123u32);
        program.set_input(&234u32);
        program.set_input(&345u32);
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace) = program.trace();

        let preprocessing = RV32IJoltVM::preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (jolt_proof, jolt_commitments, debug_info) =
            <RV32IJoltVM as Jolt<
                _,
                HyraxScheme<G1Projective, KeccakTranscript>,
                C,
                M,
                KeccakTranscript,
            >>::prove(io_device, trace, preprocessing.clone());
        let verification_result =
            RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments, debug_info);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[ignore = "Opening proof reduction for Hyrax doesn't work right now"]
    #[test]
    fn sha3_e2e_hyrax() {
        let guard = SHA3_FILE_LOCK.lock().unwrap();

        let mut program = host::Program::new("sha3-guest");
        program.set_input(&[5u8; 32]);
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace) = program.trace();
        drop(guard);

        let preprocessing = RV32IJoltVM::preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (jolt_proof, jolt_commitments, debug_info) =
            <RV32IJoltVM as Jolt<
                _,
                HyraxScheme<G1Projective, KeccakTranscript>,
                C,
                M,
                KeccakTranscript,
            >>::prove(io_device, trace, preprocessing.clone());

        let verification_result =
            RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments, debug_info);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn sha3_e2e_zeromorph() {
        let guard = SHA3_FILE_LOCK.lock().unwrap();

        let mut program = host::Program::new("sha3-guest");
        program.set_input(&[5u8; 32]);
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace) = program.trace();
        drop(guard);

        let preprocessing = RV32IJoltVM::preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (jolt_proof, jolt_commitments, debug_info) = <RV32IJoltVM as Jolt<
            _,
            Zeromorph<Bn254, KeccakTranscript>,
            C,
            M,
            KeccakTranscript,
        >>::prove(
            io_device, trace, preprocessing.clone()
        );

        let verification_result =
            RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments, debug_info);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn sha3_e2e_hyperkzg() {
        let guard = SHA3_FILE_LOCK.lock().unwrap();

        let mut program = host::Program::new("sha3-guest");
        program.set_input(&[5u8; 32]);
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace) = program.trace();
        drop(guard);

        let preprocessing = RV32IJoltVM::preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (jolt_proof, jolt_commitments, debug_info) = <RV32IJoltVM as Jolt<
            _,
            HyperKZG<Bn254, KeccakTranscript>,
            C,
            M,
            KeccakTranscript,
        >>::prove(
            io_device, trace, preprocessing.clone()
        );

        let verification_result =
            RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments, debug_info);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn memory_ops_e2e_hyperkzg() {
        let mut program = host::Program::new("memory-ops-guest");
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace) = program.trace();

        let preprocessing = RV32IJoltVM::preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (jolt_proof, jolt_commitments, debug_info) = <RV32IJoltVM as Jolt<
            _,
            HyperKZG<Bn254, KeccakTranscript>,
            C,
            M,
            KeccakTranscript,
        >>::prove(
            io_device, trace, preprocessing.clone()
        );

        let verification_result =
            RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments, debug_info);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    #[should_panic]
    fn truncated_trace() {
        let artifact_guard = FIB_FILE_LOCK.lock().unwrap();
        let mut program = host::Program::new("fibonacci-guest");
        program.set_input(&9u32);
        let (bytecode, memory_init) = program.decode();
        let (mut io_device, mut trace) = program.trace();
        trace.truncate(100);
        io_device.outputs[0] = 0; // change the output to 0
        drop(artifact_guard);

        let preprocessing = RV32IJoltVM::preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (proof, commitments, debug_info) = <RV32IJoltVM as Jolt<
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            C,
            M,
            KeccakTranscript,
        >>::prove(
            io_device, trace, preprocessing.clone()
        );
        let _verification_result =
            RV32IJoltVM::verify(preprocessing, proof, commitments, debug_info);
    }

    #[test]
    #[should_panic]
    fn malicious_trace() {
        let artifact_guard = FIB_FILE_LOCK.lock().unwrap();
        let mut program = host::Program::new("fibonacci-guest");
        program.set_input(&1u8); // change input to 1 so that termination bit equal true
        let (bytecode, memory_init) = program.decode();
        let (mut io_device, trace) = program.trace();
        let memory_layout = io_device.memory_layout.clone();
        drop(artifact_guard);

        // change memory address of output & termination bit to the same address as input
        // changes here should not be able to spoof the verifier result
        io_device.memory_layout.output_start = io_device.memory_layout.input_start;
        io_device.memory_layout.output_end = io_device.memory_layout.input_end;
        io_device.memory_layout.termination = io_device.memory_layout.input_start;

        // Since the preprocessing is done with the original memory layout, the verifier should fail
        let preprocessing = RV32IJoltVM::preprocess(
            bytecode.clone(),
            memory_layout,
            memory_init,
            1 << 20,
            1 << 20,
            1 << 20,
        );
        let (proof, commitments, debug_info) = <RV32IJoltVM as Jolt<
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            C,
            M,
            KeccakTranscript,
        >>::prove(
            io_device, trace, preprocessing.clone()
        );
        let _verification_result =
            RV32IJoltVM::verify(preprocessing, proof, commitments, debug_info);
    }
}
