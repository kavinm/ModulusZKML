use remainder::{
    layer::{matmult::Matrix, LayerId},
    mle::{Mle, MleRef},
    prover::{
        input_layer::{
            combine_input_layers::InputLayerBuilder, ligero_input_layer::LigeroInputLayer,
            InputLayer,
        },
        GKRCircuit, Layers, Witness,
    },
};
use remainder_shared_types::{
    transcript::{poseidon_transcript::PoseidonTranscript, Transcript},
    FieldExt,
};

use super::workshop_dims::{MLPInputData, MLPWeights};
use super::{
    bias_builder::BiasBuilder, bits_are_binary_builder::BitsAreBinaryBuilder,
    circuit_builders::BinaryRecompCheckerBuilder, partial_recomp_builder::BinaryRecompBuilder32Bit,
    relu_builder::ReLUBuilder,
};

pub struct MLPCircuit<F: FieldExt> {
    pub mlp_weights: MLPWeights<F>,
    pub mlp_input: MLPInputData<F>,
}

impl<F: FieldExt> GKRCircuit<F> for MLPCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- Deal with input layer shenanigans ---
        let (input_mles_input_layer, parameter_mles_input_layer, hint_mles_input_layer) =
            self.input_layer_helper();

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // **************************** BEGIN: checking the bits are binary ****************************

        // TODO: Use your code from `bits_are_binary_builder.rs` to create a check
        // that all of the values within `self.mlp_input.relu_bin_decomp` are actually
        // binary!
        let bits_are_binary_result: remainder::mle::zero::ZeroMleRef<F> = todo!();

        // **************************** BEGIN: matmul + bias ****************************
        let input_matrix = Matrix::new(
            self.mlp_input.input_mle.mle_ref(),
            1 as usize,
            self.mlp_input.input_mle.mle_ref().bookkeeping_table().len(),
            self.mlp_input.input_mle.get_prefix_bits(),
        );

        let hidden_weights_matrix = Matrix::new(
            self.mlp_weights
                .hidden_linear_weight_bias
                .weights_mle
                .mle_ref(),
            self.mlp_weights.hidden_linear_weight_bias.dim.in_features,
            self.mlp_weights.hidden_linear_weight_bias.dim.out_features,
            self.mlp_weights
                .hidden_linear_weight_bias
                .weights_mle
                .get_prefix_bits(),
        );

        let hidden_matmul_out = layers.add_matmult_layer(input_matrix, hidden_weights_matrix);

        // --- Bias circuit ---
        let bias_builder = BiasBuilder::new(
            hidden_matmul_out,
            self.mlp_weights
                .hidden_linear_weight_bias
                .biases_mle
                .clone(),
        );
        let hidden_matmul_bias_out = layers.add_gkr(bias_builder);

        // **************************** BEGIN: ReLU ****************************
        // --- Check that the bin decomp is correct with respect to the current outputs ---
        let pos_bin_recomp_builder =
            BinaryRecompBuilder32Bit::new(self.mlp_input.relu_bin_decomp.clone());
        let pos_bin_recomp_mle = layers.add_gkr(pos_bin_recomp_builder);
        let recomp_checker_builder = BinaryRecompCheckerBuilder::new(
            hidden_matmul_bias_out.clone(),
            self.mlp_input.relu_bin_decomp.clone(),
            pos_bin_recomp_mle.clone(),
        );
        let recomp_checker_result = layers.add_gkr(recomp_checker_builder);

        // --- Finally, the actual ReLU circuit ---
        // TODO: Use the code within `relu_builder` to instantiate a builder which
        // computes ReLU(`hidden_matmul_bias_out`) and sets the output to `relu_result`!
        let relu_result: remainder::mle::dense::DenseMle<F, F> = todo!();

        // **************************** BEGIN: output layer matmul + bias ****************************
        // --- Finally, compute the final matmul/bias layer, without ReLU ---
        let final_hidden_layer_vals_matrix = Matrix::new_with_padding(
            relu_result.mle_ref(),
            1 as usize,
            relu_result.mle_ref().bookkeeping_table().len(),
            relu_result.get_prefix_bits(),
        );

        let out_linear_weight_bias = self.mlp_weights.out_linear_weight_bias.clone();
        let out_weights_matrix = Matrix::new(
            out_linear_weight_bias.weights_mle.mle_ref(),
            out_linear_weight_bias.dim.in_features,
            out_linear_weight_bias.dim.out_features,
            out_linear_weight_bias.weights_mle.get_prefix_bits(),
        );

        let final_matmul_out =
            layers.add_matmult_layer(final_hidden_layer_vals_matrix, out_weights_matrix);

        // --- Bias circuit ---
        let final_bias_builder =
            BiasBuilder::new(final_matmul_out, out_linear_weight_bias.clone().biases_mle);
        let final_matmul_bias_out = layers.add_gkr(final_bias_builder);

        // --- Output layers include bits are binary check, recomp check, and the final result ---
        let output_layers: Vec<remainder::mle::mle_enum::MleEnum<F>> = vec![
            bits_are_binary_result.get_enum(),
            recomp_checker_result.get_enum(),
            final_matmul_bias_out.mle_ref().get_enum(),
        ];

        Witness {
            layers,
            output_layers,
            input_layers: vec![
                input_mles_input_layer.to_enum(),
                parameter_mles_input_layer.to_enum(),
                hint_mles_input_layer.to_enum(),
            ],
        }
    }
}

impl<F: FieldExt> MLPCircuit<F> {
    /// Creates a new instance of BinDecomp16BitsAreBinaryCircuit
    pub fn new(mlp_weights: MLPWeights<F>, mlp_input: MLPInputData<F>) -> Self {
        Self {
            mlp_weights,
            mlp_input,
        }
    }

    /// Literally just to make the actual code look a little nicer
    pub fn input_layer_helper(
        &mut self,
    ) -> (
        LigeroInputLayer<F, PoseidonTranscript<F>>,
        LigeroInputLayer<F, PoseidonTranscript<F>>,
        LigeroInputLayer<F, PoseidonTranscript<F>>,
    ) {
        const MODEL_INPUT_IDX: usize = 0;
        const MODEL_PARAM_IDX: usize = 1;
        const MODEL_HINTS_IDX: usize = 2;

        // --- Adjust layer IDs for associated input layer MLEs ---
        self.mlp_input.input_mle.layer_id = LayerId::Input(MODEL_INPUT_IDX);

        self.mlp_weights
            .hidden_linear_weight_bias
            .weights_mle
            .layer_id = LayerId::Input(MODEL_PARAM_IDX);
        self.mlp_weights
            .hidden_linear_weight_bias
            .biases_mle
            .layer_id = LayerId::Input(MODEL_PARAM_IDX);
        self.mlp_weights.out_linear_weight_bias.weights_mle.layer_id =
            LayerId::Input(MODEL_PARAM_IDX);
        self.mlp_weights.out_linear_weight_bias.biases_mle.layer_id =
            LayerId::Input(MODEL_PARAM_IDX);

        self.mlp_input.relu_bin_decomp.layer_id = LayerId::Input(MODEL_HINTS_IDX);

        // --- Inputs to the circuit include model inputs, parameters, and hints ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.mlp_input.input_mle as &mut dyn Mle<F>)];
        let parameter_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut self.mlp_weights.hidden_linear_weight_bias.weights_mle),
            Box::new(&mut self.mlp_weights.hidden_linear_weight_bias.biases_mle),
            Box::new(&mut self.mlp_weights.out_linear_weight_bias.weights_mle),
            Box::new(&mut self.mlp_weights.out_linear_weight_bias.biases_mle),
        ];
        let hint_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.mlp_input.relu_bin_decomp)];

        // --- Create input layers ---
        let input_mles_input_layer_builder =
            InputLayerBuilder::new(input_mles, None, LayerId::Input(MODEL_INPUT_IDX));
        let input_mles_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> =
            input_mles_input_layer_builder.to_input_layer_with_rho_inv(4_u8, 1_f64);

        let parameter_mles_input_layer_builder =
            InputLayerBuilder::new(parameter_mles, None, LayerId::Input(MODEL_PARAM_IDX));
        let parameter_mles_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> =
            parameter_mles_input_layer_builder.to_input_layer_with_rho_inv(4_u8, 1_f64);

        let hint_mles_input_layer_builder =
            InputLayerBuilder::new(hint_mles, None, LayerId::Input(MODEL_HINTS_IDX));
        let hint_mles_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> =
            hint_mles_input_layer_builder.to_input_layer_with_rho_inv(4_u8, 1_f64);

        (
            input_mles_input_layer,
            parameter_mles_input_layer,
            hint_mles_input_layer,
        )
    }
}

// #[ignore]
#[test]
fn test_full_circuit() {
    use super::workshop_utils::load_dummy_mlp_input_and_weights;
    use remainder_shared_types::Fr;

    let (mnist_weights, mnist_inputs) = load_dummy_mlp_input_and_weights::<Fr>(16, 4, 4);

    let mut circuit = MLPCircuit::<Fr>::new(mnist_weights, mnist_inputs);

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
        }
        Err(err) => {
            println!("{}", err);
            panic!();
        }
    }
}
