use ark_std::log2;
use itertools::{repeat_n, Itertools};
use remainder::{layer::{batched::{combine_zero_mle_ref, BatchedLayer}, matmult::Matrix, LayerId}, mle::{bin_decomp_structs::bin_decomp_64_bit::BinDecomp64Bit, dense::DenseMle, Mle, MleIndex, MleRef}, prover::{input_layer::{combine_input_layers::InputLayerBuilder, ligero_input_layer::LigeroInputLayer, InputLayer}, GKRCircuit, Layers, Witness}};
use remainder::prover::input_layer::public_input_layer::PublicInputLayer;
use remainder_shared_types::{transcript::{poseidon_transcript::PoseidonTranscript, Transcript}, FieldExt};

use crate::dims::{DataparallelMLPInputData, MLPWeights};
use crate::dataparallel_circuit::{dataparallel_bias_builder::BiasBuilder, dataparallel_bits_are_binary_builder::DataparallelBitsAreBinaryBuilder, dataparallel_circuit_builders::{BinaryRecompCheckerBuilder, PositiveBinaryRecompBuilder}, dataparallel_partial_recomp_builder::PartialPositiveBinaryRecompBuilder, dataparallel_relu_builder::ReLUBuilder};
use crate::dims::MLPInputData;
pub struct DataparallelMLPCircuit<F: FieldExt> {
    pub mlp_weights: MLPWeights<F>,
    pub mlp_input: DataparallelMLPInputData<F>,
}

impl<F: FieldExt> GKRCircuit<F> for DataparallelMLPCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        const RECOMP_BITWIDTH: usize = 32;

        // --- Inputs to the circuit include model inputs, weights, biases, and decomps ---
        // let mut combined_input_mles = DenseMle::<F, F>::combine_mle_batch(self.mlp_input.input_mles.clone());
        
        let mut combined_input_mles = DenseMle::batch_dense_mle(self.mlp_input.input_mles.clone());
        let box_combined_input_mles: Box<&mut dyn Mle<F>> = Box::new(&mut combined_input_mles);

        let (all_weights_mles, all_biases_mles): (Vec<Box<&mut dyn Mle<F>>>, Vec<Box<&mut dyn Mle<F>>>) = self.mlp_weights.all_linear_weights_biases.iter_mut().map(|linear_weight_bias| {
            (Box::new(&mut linear_weight_bias.weights_mle as &mut dyn Mle<F>), Box::new(&mut linear_weight_bias.biases_mle as &mut dyn Mle<F>))
        }).unzip();

        // --- All decomps ---
        let mut relu_bin_decomp_combined_vecs = self.mlp_input.relu_bin_decomp_vecs.clone().into_iter().map(|relu_bin_decomp_vec| {
            DenseMle::<F, BinDecomp64Bit<F>>::combine_mle_batch(relu_bin_decomp_vec)
        }).collect_vec();

        // --- Put it altogether ---
        let circuit_input_mles: Vec<Box<&mut dyn Mle<F>>> = std::iter::once(box_combined_input_mles)
        .chain(all_weights_mles)
        .chain(all_biases_mles)
        .chain(
            relu_bin_decomp_combined_vecs.iter_mut().map(|bin_decomp_mle_combined| {
                Box::new(bin_decomp_mle_combined as &mut dyn Mle<F>)
            }).collect_vec()
        ).collect_vec();

        let input_layer_builder = InputLayerBuilder::new(circuit_input_mles, None, LayerId::Input(0));

        // --- Create input layers ---
        let live_committed_input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer_with_rho_inv(4_u8, 1_f64);
        
        // println!("the prefix for box_combined_input_mles {:?}", combined_input_mles.get_prefix_bits());
        // println!("the prefix for all_weights_mles {:?}", self.mlp_weights.all_linear_weights_biases[0].weights_mle.get_prefix_bits());
        // println!("the prefix for all_weights_mles {:?}", self.mlp_weights.all_linear_weights_biases[1].weights_mle.get_prefix_bits());
        // println!("the prefix for all_biases_mles {:?}", self.mlp_weights.all_linear_weights_biases[0].biases_mle.get_prefix_bits());
        // println!("the prefix for all_biases_mles {:?}", self.mlp_weights.all_linear_weights_biases[1].biases_mle.get_prefix_bits());
        // println!("the prefix for box_combined_input_mles {:?}", relu_bin_decomp_combined_vecs[0].get_prefix_bits());

        // set the batched input / relu decomp's prefix bits
        self.mlp_input.input_mles.iter_mut().for_each(|mle| {
            mle.set_prefix_bits(combined_input_mles.get_prefix_bits());
        });
        self.mlp_input.relu_bin_decomp_vecs.iter_mut().enumerate().map(|(relu_idx, relu_bin_decomp_vec)| {
            relu_bin_decomp_vec.iter_mut().for_each(|relu_bin_decomp| {

                println!("relu_bin_decomp.get_prefix_bits() {:?}", relu_bin_decomp_combined_vecs[relu_idx].get_prefix_bits());
                relu_bin_decomp.set_prefix_bits(relu_bin_decomp_combined_vecs[relu_idx].get_prefix_bits());
            });
        }).collect_vec();

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Dataparallel stuff ---
        assert_eq!(self.mlp_input.input_mles.len(), self.mlp_input.dim.batch_size);
        let num_dataparallel_bits = log2(self.mlp_input.dim.batch_size) as usize;

        println!("num_dataparallel_bits: {:?}", num_dataparallel_bits);

        println!("num of layers {:?}", self.mlp_input.relu_bin_decomp_vecs.len());

        // **************************** BEGIN: checking the bits are binary ****************************

        // --- Create the builders for (b_i)^2 - b_i (but for all layers) ---

        // --- Unfortunate hacky way to avoid closure to avoid borrowing `layers` as mutable multiple times in the same scope ---
        let mut bits_are_binary_zero_mle_refs = vec![];
        for bin_decomp_mle_vec in self.mlp_input.relu_bin_decomp_vecs.clone() {
            let bits_are_binary_builders = bin_decomp_mle_vec
                .into_iter().map(
                    |mut bin_decomp_mle| {
                        bin_decomp_mle.add_batch_bits(num_dataparallel_bits);
                        DataparallelBitsAreBinaryBuilder::new(bin_decomp_mle) // ad hoc
                    }
                ).collect_vec();
            let batched_bits_are_binary_builders = BatchedLayer::new(bits_are_binary_builders);
            let bits_are_binary_result = layers.add_gkr(batched_bits_are_binary_builders);
            let circuit_output = combine_zero_mle_ref(bits_are_binary_result);
            bits_are_binary_zero_mle_refs.push(circuit_output);
        }

        // **************************** BEGIN: matmul + bias, then ReLU ****************************
        let (recomp_checker_zero_mle_refs, penultimate_hidden_layer_mle) = self.mlp_weights.all_linear_weights_biases
            .iter()
            // --- Skip the last one (we don't want ReLU there) ---
            .rev()
            .skip(1)
            .rev()
            // ----------------------------------------------------
            .zip(self.mlp_input.relu_bin_decomp_vecs.clone().into_iter())
            .fold((vec![], self.mlp_input.input_mles.clone()), 
            |(recomp_checker_zero_mle_refs_acc, last_hidden_layer_vals_mle), (linear_weight_bias, relu_bin_decomp)| {

                // --- Construct input/weight matrices and compute matmul ---
                let hidden_layer_vals_matrix = Matrix::new(
                    last_hidden_layer_vals_mle[0].mle_ref(), // ad hoc
                    1 as usize,
                    last_hidden_layer_vals_mle[0].mle_ref().bookkeeping_table().len(),
                    last_hidden_layer_vals_mle[0].get_prefix_bits(),
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
                let partial_pos_bin_recomp_builder = PartialPositiveBinaryRecompBuilder::new(
                    relu_bin_decomp[0].clone(), // ad hoc
                    RECOMP_BITWIDTH
                );
                let partial_pos_bin_recomp_mle = layers.add_gkr(partial_pos_bin_recomp_builder);

                // --- Next, the recomp checker ---
                let recomp_checker_builder = BinaryRecompCheckerBuilder::new(
                    matmul_bias_out.clone(),
                    relu_bin_decomp[0].clone(),
                );
                let recomp_checker_result = layers.add_gkr(recomp_checker_builder);

                // --- Finally, the actual ReLU circuit ---
                let relu_builder = ReLUBuilder::new(relu_bin_decomp[0].clone(), partial_pos_bin_recomp_mle);
                let relu_result = layers.add_gkr(relu_builder);

                let new_recomp_checker_zero_mle_refs_acc = recomp_checker_zero_mle_refs_acc
                    .into_iter()
                    .chain(std::iter::once(recomp_checker_result))
                    .collect_vec();

                // --- Updated ZeroMLERefs, updated hidden layer state ---
                (new_recomp_checker_zero_mle_refs_acc, vec![relu_result]) // ad hoc
        });

        println!("recomp_checker_zero_mle_refs {:?}", recomp_checker_zero_mle_refs.len());

        let last_linear_weight_bias = self.mlp_weights.all_linear_weights_biases.last().unwrap();
        let batched_penultimate_hidden_layer_mle = DenseMle::batch_dense_mle(penultimate_hidden_layer_mle);

        // --- Finally, compute the final matmul/bias layer, without ReLU ---
        let final_hidden_layer_vals_matrix = Matrix::new(
            batched_penultimate_hidden_layer_mle.mle_ref(), // ad hoc
            1 << num_dataparallel_bits as usize,
            last_linear_weight_bias.dim.in_features,
            batched_penultimate_hidden_layer_mle.get_prefix_bits(),
        );

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

        // println!("final_matmul_bias_out {:?}", final_matmul_bias_out);

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
        mlp_input: DataparallelMLPInputData<F>,
    ) -> Self {
        Self {
            mlp_weights,
            mlp_input,
        }
    }
}
// #[ignore]
#[test]
fn test_full_circuit_mlp() {

    // use crate::dims::{NNLinearDimension, NNLinearInputDimension};
    use crate::utils::load_batched_dummy_mlp_input_and_weights;
    use remainder_shared_types::Fr;

    let (mnist_weights, mnist_inputs) = load_batched_dummy_mlp_input_and_weights::<Fr>(
        16,
        Some(vec![]),
        // None,
        4,
        4,
    );

    let mut circuit = DataparallelMLPCircuit::<Fr>::new(
        mnist_weights, mnist_inputs,
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