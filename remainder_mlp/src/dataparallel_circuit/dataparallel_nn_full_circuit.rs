use ark_std::log2;
use itertools::Itertools;
use remainder::{layer::{matmult::Matrix, LayerId}, mle::{bin_decomp_structs::bin_decomp_64_bit::BinDecomp64Bit, dense::DenseMle, Mle, MleRef}, prover::{input_layer::{combine_input_layers::InputLayerBuilder, ligero_input_layer::LigeroInputLayer, InputLayer}, GKRCircuit, Layers, Witness}};
use remainder_shared_types::{transcript::{poseidon_transcript::PoseidonTranscript, Transcript}, FieldExt};

use crate::dims::{DataparallelMLPInputData, MLPWeights};

use super::dataparallel_bits_are_binary_builder::DataparallelBitsAreBinaryBuilder;

pub struct DataparallelMLPCircuit<F: FieldExt> {
    pub mlp_weights: MLPWeights<F>,
    pub mlp_input: DataparallelMLPInputData<F>,
}

impl<F: FieldExt> GKRCircuit<F> for DataparallelMLPCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- Inputs to the circuit include model inputs, weights, biases, and decomps ---
        let combined_input_mles = DenseMle::<F, F>::combine_mle_batch(self.mlp_input.input_mles.clone());
        let box_combined_input_mles: Box<&mut dyn Mle<F>> = Box::new(&mut combined_input_mles);

        let (all_weights_mles, all_biases_mles): (Vec<Box<&mut dyn Mle<F>>>, Vec<Box<&mut dyn Mle<F>>>) = self.mlp_weights.all_linear_weights_biases.iter_mut().map(|linear_weight_bias| {
            (Box::new(&mut linear_weight_bias.weights_mle as &mut dyn Mle<F>), Box::new(&mut linear_weight_bias.biases_mle as &mut dyn Mle<F>))
        }).unzip();

        // --- Put it altogether ---
        let circuit_input_mles: Vec<Box<&mut dyn Mle<F>>> = std::iter::once(box_combined_input_mles)
        .chain(all_weights_mles)
        .chain(all_biases_mles)
        .chain(
            // --- All decomps ---
            self.mlp_input.relu_bin_decomp.into_iter().map(|relu_bin_decomp| {
                let combined_relu_bin_decomp = &mut DenseMle::<F, BinDecomp64Bit<F>>::combine_mle_batch(relu_bin_decomp);
                Box::new(combined_relu_bin_decomp as &mut dyn Mle<F>)
            })
        ).collect_vec();

        let input_layer_builder = InputLayerBuilder::new(circuit_input_mles, None, LayerId::Input(0));

        // --- Create input layers ---
        let live_committed_input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer_with_rho_inv(4_u8, 1_f64);

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Dataparallel stuff ---
        let num_dataparallel_bits = log2(self.mlp_input.input_mles.len()) as usize;

        // **************************** BEGIN: checking the bits are binary ****************************

        // --- Unfortunate hacky way to avoid closure to avoid borrowing `layers` as mutable multiple times in the same scope ---
        let mut bits_are_binary_zero_mle_refs = vec![];
        for bin_decomp_mle in self.mlp_input.relu_bin_decomp.clone() {
            let bits_are_binary_builder = DataparallelBitsAreBinaryBuilder::new(bin_decomp_mle);
            let bits_are_binary_result = layers.add_gkr(bits_are_binary_builder);
            bits_are_binary_zero_mle_refs.push(bits_are_binary_result);
        }

        // **************************** BEGIN: matmul + bias, then ReLU ****************************
        let (recomp_checker_zero_mle_refs, penultimate_hidden_layer_mle) = self.mlp_weights.all_linear_weights_biases
            .iter()
            // --- Skip the last one (we don't want ReLU there) ---
            .rev()
            .skip(1)
            .rev()
            // ----------------------------------------------------
            .zip(self.mlp_input.relu_bin_decomp.clone().into_iter())
            .fold((vec![], self.mlp_input.input_mle.clone()), 
            |(recomp_checker_zero_mle_refs_acc, last_hidden_layer_vals_mle), (linear_weight_bias, relu_bin_decomp)| {

                // --- Construct input/weight matrices and compute matmul ---
                let hidden_layer_vals_matrix = Matrix::new(
                    last_hidden_layer_vals_mle.mle_ref(),
                    1 as usize,
                    last_hidden_layer_vals_mle.mle_ref().bookkeeping_table().len(),
                    last_hidden_layer_vals_mle.get_prefix_bits(),
                );

                let weights_matrix = Matrix::new(
                    linear_weight_bias.weights_mle.mle_ref(),
                    linear_weight_bias.dim.in_features,
                    linear_weight_bias.dim.out_features,
                    linear_weight_bias.weights_mle.get_prefix_bits(),
                );

                let matmul_out = layers.add_matmult_layer(hidden_layer_vals_matrix, weights_matrix);
                // dbg!(matmul_out.num_iterated_vars());
                // dbg!(matmul_out.mle_ref().bookkeeping_table().len());

                // --- Bias circuit ---
                let bias_builder = BiasBuilder::new(matmul_out, linear_weight_bias.biases_mle.clone());
                let matmul_bias_out = layers.add_gkr(bias_builder);
                // dbg!(matmul_bias_out.num_iterated_vars());
                // dbg!(matmul_bias_out.mle_ref().bookkeeping_table().len());

                // --- Check that the bin decomp is correct with respect to the current outputs ---
                let partial_pos_bin_recomp_builder = PartialPositiveBinaryRecompBuilder::new(relu_bin_decomp.clone(), 16);
                let partial_pos_bin_recomp_mle = layers.add_gkr(partial_pos_bin_recomp_builder);

                // --- Next, the recomp checker ---
                let recomp_checker_builder = BinaryRecompCheckerBuilder::new(
                    matmul_bias_out.clone(),
                    relu_bin_decomp.clone(),
                    partial_pos_bin_recomp_mle,
                );
                let recomp_checker_result = layers.add_gkr(recomp_checker_builder);

                // --- Finally, the actual ReLU circuit ---
                let relu_builder = ReLUBuilder::new(relu_bin_decomp, matmul_bias_out);
                let relu_result = layers.add_gkr(relu_builder);

                let new_recomp_checker_zero_mle_refs_acc = recomp_checker_zero_mle_refs_acc
                    .into_iter()
                    .chain(std::iter::once(recomp_checker_result))
                    .collect_vec();

                // --- Updated ZeroMLERefs, updated hidden layer state ---
                (new_recomp_checker_zero_mle_refs_acc, relu_result)
        });

        // --- Finally, compute the final matmul/bias layer, without ReLU ---
        let final_hidden_layer_vals_matrix = Matrix::new(
            penultimate_hidden_layer_mle.mle_ref(),
            1 as usize,
            penultimate_hidden_layer_mle.mle_ref().bookkeeping_table().len(),
            penultimate_hidden_layer_mle.get_prefix_bits(),
        );

        let last_linear_weight_bias = self.mlp_weights.all_linear_weights_biases.last().unwrap();
        let final_weights_matrix = Matrix::new(
            last_linear_weight_bias.weights_mle.mle_ref(),
            last_linear_weight_bias.dim.in_features,
            last_linear_weight_bias.dim.out_features,
            last_linear_weight_bias.weights_mle.get_prefix_bits(),
        );

        let final_matmul_out = layers.add_matmult_layer(final_hidden_layer_vals_matrix, final_weights_matrix);

        // --- Bias circuit ---
        let final_bias_builder = BiasBuilder::new(final_matmul_out, last_linear_weight_bias.clone().biases_mle);
        let final_matmul_bias_out = layers.add_gkr(final_bias_builder);

        // --- Output layers include bits are binary checks, recomp checks, and the final result ---
        let output_layers = std::iter::once(final_matmul_bias_out.mle_ref().get_enum())
            .chain(
                recomp_checker_zero_mle_refs.into_iter().map(|zero_mle_ref| zero_mle_ref.get_enum())
            )
            .chain(
                bits_are_binary_zero_mle_refs.into_iter().map(|zero_mle_ref| zero_mle_ref.get_enum())
            ).collect_vec();
            // .collect_vec();

        Witness {
            layers,
            output_layers,
            input_layers: vec![live_committed_input_layer.to_enum()]
        }
    }
}


impl<F: FieldExt> DataparallelMLPCircuit<F> {
    /// Creates a new instance of DataparallelMLPCircuit
    pub fn new(
        mlp_weights: MLPWeights<F>,
        mlp_input: MLPInputData<F>,
    ) -> Self {
        Self {
            mlp_weights,
            mlp_input,
        }
    }
}
// #[ignore]
#[test]
fn test_full_circuit() {

    // use crate::dims::{NNLinearDimension, NNLinearInputDimension};
    use crate::utils::load_dummy_mlp_input_and_weights;
    use remainder_shared_types::Fr;

    let (mnist_weights, mnist_inputs) = load_dummy_mlp_input_and_weights::<Fr>(
        16,
        Some(vec![4]),
        // None,
        4,
    );

    let mut circuit = DataparallelMLPCircuit::<Fr>::new(
        mnist_weights, mnist_inputs, 2,
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