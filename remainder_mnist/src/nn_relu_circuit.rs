use ark_std::log2;
use itertools::{repeat_n, Itertools};
use remainder::{expression::ExpressionStandard, layer::{batched::{combine_mles, combine_zero_mle_ref, BatchedLayer}, from_mle, LayerId}, mle::{dense::DenseMle, structs::BinDecomp16Bit, zero::ZeroMleRef, Mle, MleIndex, MleRef}, prover::{input_layer::{combine_input_layers::InputLayerBuilder, ligero_input_layer::LigeroInputLayer, public_input_layer::PublicInputLayer, InputLayer}, GKRCircuit, Layers, Witness}};
use remainder_shared_types::{transcript::{poseidon_transcript::PoseidonTranscript, Transcript}, FieldExt, Fr};

use crate::{bits_are_binary_builder::BitsAreBinaryBuilder, circuit_builders::{BinaryRecompCheckerBuilder, PositiveBinaryRecompBuilder}, relu_builder::ReLUBuilder, self_subtract_builder::SelfSubtractBuilder, utils::generate_16_bit_decomp_signed};

pub struct ReluCircuit<F: FieldExt> {
    pub relu_in: DenseMle<F, F>,
    pub signed_bin_decomp_mle: DenseMle<F, BinDecomp16Bit<F>>,
}

impl<F: FieldExt> GKRCircuit<F> for ReluCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- For the input layer, we need to first merge all of the input MLEs FIRST by mle_idx, then by dataparallel index ---
        // --- This assures that (going left-to-right in terms of the bits) we have [input_prefix_bits], [dataparallel_bits], [mle_idx], [iterated_bits] ---
        // let mut combined_mles = DenseMle::<F, F>::combine_mle_batch(self.mles.clone());
        // let mut combined_signed_bin_decomp_mles = DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(self.signed_bin_decomp_mles.clone());

        // --- Inputs to the circuit are just these two MLEs ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.relu_in), Box::new(&mut self.signed_bin_decomp_mle)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, Some(vec![self.relu_in.num_iterated_vars()]), LayerId::Input(0));

        // --- Create input layers ---
        let live_committed_input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer_with_rho_inv(4_u8, 1_f64);

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // **************************** BEGIN: checking the bits recompute to the mles ****************************

        // --- First we create the positive binary recomp builder ---
        let pos_bin_recomp_builder = PositiveBinaryRecompBuilder::new(self.signed_bin_decomp_mle.clone());
        let pos_bin_recomp_mle = layers.add_gkr(pos_bin_recomp_builder);

        // --- Finally, the recomp checker ---
        let recomp_checker_builder = BinaryRecompCheckerBuilder::new(
            self.relu_in.clone(),
            self.signed_bin_decomp_mle.clone(),
            pos_bin_recomp_mle,
        );

        // --- Grab output layer and flatten ---
        let recomp_checker_result = layers.add_gkr(recomp_checker_builder);

        // **************************** END: checking the bits recompute to the mles ****************************

        // **************************** BEGIN: checking the bits are binary ****************************

        // --- Create the builders for (b_i)^2 - b_i ---
        let bits_are_binary_builder = BitsAreBinaryBuilder::new(self.signed_bin_decomp_mle);
        let bits_are_binary_result = layers.add_gkr(bits_are_binary_builder);

        // **************************** END: checking the bits are binary ****************************

        // **************************** BEGIN: the actual relu circuit ****************************

        // --- Create the builders for (1 - b_i) * x_i ---
        // ---   this simplifies to x_i - b_i * x_i    ---
        let relu_builder = ReLUBuilder::new(self.signed_bin_decomp_mle, self.relu_in);
        let relu_result = layers.add_gkr(relu_builder);

        // --- Finally, need to subtract relu result from itself to get ZeroMleRef lol ---
        let self_sub_builder = SelfSubtractBuilder::new(relu_result);
        let final_result = layers.add_gkr(self_sub_builder);

        // **************************** END: the actual relu circuit ****************************

        Witness {
            layers,
            output_layers: vec![
                recomp_checker_result.get_enum(),
                bits_are_binary_result.get_enum(),
                final_result.get_enum(),
            ],
            input_layers: vec![live_committed_input_layer.to_enum()]
        }
    }
}


impl<F: FieldExt> ReluCircuit<F> {
    /// Creates a new instance of BinDecomp16BitsAreBinaryCircuit
    pub fn new(
        mles: DenseMle<F, F>,
        signed_bin_decomp_mles: DenseMle<F, BinDecomp16Bit<F>>,
    ) -> Self {
        Self {
            relu_in: mles,
            signed_bin_decomp_mle: signed_bin_decomp_mles,
        }
    }
}

#[test]
fn test_relu_circuit() {

    // generate a random 16 bits, compute the number from the decomposition
    let (
        binary_decomp_mle_vec,
        binary_recomp_mle_vec
    ) = generate_16_bit_decomp_signed::<Fr>(2);

    let mut circuit = ReluCircuit::new(
        binary_recomp_mle_vec,
        binary_decomp_mle_vec,
    );

    let mut transcript = PoseidonTranscript::new("Relu Circuit Transcript");
    let proof = circuit.prove(&mut transcript);

    match proof {
        Ok(proof) => {
            let mut transcript = PoseidonTranscript::new("Relu Circuit Transcript");
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