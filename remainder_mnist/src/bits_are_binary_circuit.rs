use ark_ff::Field;
use ark_std::{log2, test_rng};
use itertools::{repeat_n, Itertools};
use rand::Rng;
use remainder::{expression::ExpressionStandard, layer::{batched::{combine_zero_mle_ref, BatchedLayer}, from_mle, LayerId}, mle::{dense::DenseMle, structs::BinDecomp16Bit, zero::ZeroMleRef, Mle, MleIndex, MleRef}, prover::{input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer}, GKRCircuit, Layers, Witness}};
use remainder_shared_types::{transcript::{poseidon_transcript::PoseidonTranscript, Transcript}, FieldExt, Fr};

use crate::utils::generate_16_bit_decomp;

/// Checks that all of the bits within a `BinDecomp16Bit` are indeed binary
/// via b_i^2 - b_i = 0 (but it's batched)
pub struct BinDecomp16BitsAreBinaryCircuit<F: FieldExt> {
    batched_diff_signed_bin_decomp_mle: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinDecomp16BitsAreBinaryCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- Input to the circuit is just the one (combined) MLE ---
        let mut combined_batched_diff_signed_bin_decomp_mle = DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(self.batched_diff_signed_bin_decomp_mle.clone());
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut combined_batched_diff_signed_bin_decomp_mle)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let live_committed_input_layer: PublicInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer();

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Dataparallel/batching stuff + sanitychecks ---
        let num_subcircuit_copies = self.batched_diff_signed_bin_decomp_mle.len();
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;

        // --- Create the builders for (b_i)^2 - b_i ---
        let diff_builders = self.batched_diff_signed_bin_decomp_mle.iter_mut().map(|diff_signed_bin_decomp_mle| {
            diff_signed_bin_decomp_mle.set_prefix_bits(
                Some(
                    diff_signed_bin_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
                        combined_batched_diff_signed_bin_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
                            repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                        )
                    ).collect_vec()
                )
            );
            
            from_mle(
                diff_signed_bin_decomp_mle, 
                |diff_signed_bin_decomp_mle| {
                    let combined_bin_decomp_mle_ref = diff_signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
                    ExpressionStandard::Product(vec![combined_bin_decomp_mle_ref.clone(), combined_bin_decomp_mle_ref.clone()]) - ExpressionStandard::Mle(combined_bin_decomp_mle_ref)
                }, 
                |mle, id, prefix_bits| {
                    ZeroMleRef::new(mle.num_iterated_vars(), prefix_bits, id)
            })
        }).collect_vec();
        let combined_output_zero_mle_ref = combine_zero_mle_ref(layers.add_gkr(BatchedLayer::new(diff_builders)));

        Witness { layers, output_layers: vec![combined_output_zero_mle_ref.get_enum()], input_layers: vec![live_committed_input_layer.to_enum()] }
    }
}

impl<F: FieldExt> BinDecomp16BitsAreBinaryCircuit<F> {
    /// Creates a new instance of BinDecomp16BitsAreBinaryCircuit
    pub fn new(
        batched_diff_signed_bin_decomp_mle: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    ) -> Self {
        Self {
            batched_diff_signed_bin_decomp_mle,
        }
    }

    /// This does exactly the same thing as `synthesize()` above, but
    /// takes in prefix bits for each of the input layer MLEs as opposed
    /// to synthesizing its own input layer.
    pub fn yield_sub_circuit(&mut self) -> Witness<F, <BinDecomp16BitsAreBinaryCircuit<F> as GKRCircuit<F>>::Transcript> {

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, <BinDecomp16BitsAreBinaryCircuit<F> as GKRCircuit<F>>::Transcript> = Layers::new();

        // --- Dataparallel/batching stuff + sanitychecks ---
        let num_subcircuit_copies = self.batched_diff_signed_bin_decomp_mle.len();
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;

        // --- Create the builders for (b_i)^2 - b_i ---
        let diff_builders = self.batched_diff_signed_bin_decomp_mle.iter_mut().map(|diff_signed_bin_decomp_mle| {
            diff_signed_bin_decomp_mle.set_prefix_bits(
                Some(
                    diff_signed_bin_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                    ).collect_vec()
                )
            );
            
            from_mle(
                diff_signed_bin_decomp_mle, 
                |diff_signed_bin_decomp_mle| {
                    let combined_bin_decomp_mle_ref = diff_signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
                    ExpressionStandard::Product(vec![combined_bin_decomp_mle_ref.clone(), combined_bin_decomp_mle_ref.clone()]) - ExpressionStandard::Mle(combined_bin_decomp_mle_ref)
                }, 
                |mle, id, prefix_bits| {
                    ZeroMleRef::new(mle.num_iterated_vars(), prefix_bits, id)
            })
        }).collect_vec();
        let combined_output_zero_mle_ref = combine_zero_mle_ref(layers.add_gkr(BatchedLayer::new(diff_builders)));
        println!("# layers -- bits r binary 16bit: {:?}", layers.next_layer_id());

        Witness { layers, output_layers: vec![combined_output_zero_mle_ref.get_enum()], input_layers: vec![] }
        
    }
}

#[test]
fn test_bin_decomp_16_bit_bits_are_binary() {

    // generate a random 16 bits, compute the number from the decomposition
    let (
        binary_decomp_mle_vec,
        _binary_recomp_mle_vec
    ) = generate_16_bit_decomp::<Fr>(4, 2);

    let mut circuit = BinDecomp16BitsAreBinaryCircuit::new(
        binary_decomp_mle_vec,
    );

    let mut transcript = PoseidonTranscript::new("Bin Decomp Bits Are Binary Transcript");
    let proof = circuit.prove(&mut transcript);

    match proof {
        Ok(proof) => {
            let mut transcript = PoseidonTranscript::new("Bin Decomp Bits Are Binary Transcript");
            let result = circuit.verify(&mut transcript, proof);
            if let Err(err) = result {
                println!("{}", err);
                panic!();
            }
        },
        Err(err) => {
            println!("{}", err);
            panic!();
        }
    }
}