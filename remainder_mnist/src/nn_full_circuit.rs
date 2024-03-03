use ark_std::log2;
use itertools::{repeat_n, Itertools};
use remainder::{expression::ExpressionStandard, layer::{batched::{combine_mles, combine_zero_mle_ref, BatchedLayer}, from_mle, matmult::Matrix, LayerId}, mle::{dense::DenseMle, structs::BinDecomp16Bit, zero::ZeroMleRef, Mle, MleIndex, MleRef}, prover::{input_layer::{combine_input_layers::InputLayerBuilder, ligero_input_layer::LigeroInputLayer, public_input_layer::PublicInputLayer, InputLayer}, GKRCircuit, Layers, Witness}};
use remainder_shared_types::{transcript::{poseidon_transcript::PoseidonTranscript, Transcript}, FieldExt, Fr};

use crate::{bias_builder::BiasBuilder, bits_are_binary_builder::BitsAreBinaryBuilder, circuit_builders::{BinaryRecompCheckerBuilder, PositiveBinaryRecompBuilder}, data_pipeline::{MNISTInputData, MNISTWeights, NNLinearDimension, NNLinearInputDimension}, relu_builder::ReLUBuilder, self_subtract_builder::SelfSubtractBuilder, utils::{generate_16_bit_decomp_signed, load_dummy_mnist_input_and_weights}};

pub struct MNISTModelCircuit<F: FieldExt> {
    pub mnist_weights: MNISTWeights<F>,
    pub mnist_input: MNISTInputData<F>,
}

impl<F: FieldExt> GKRCircuit<F> for MNISTModelCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- Inputs to the circuit are just these two MLEs ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![

            // --- For inputs ---
            Box::new(&mut self.mnist_input.input_mle),
            Box::new(&mut self.mnist_input.relu_bin_decomp),

            // --- For weights ---
            Box::new(&mut self.mnist_weights.l1_linear_weights.weights_mle),
            Box::new(&mut self.mnist_weights.l1_linear_weights.biases_mle),
            Box::new(&mut self.mnist_weights.l2_linear_weights.weights_mle),
            Box::new(&mut self.mnist_weights.l2_linear_weights.biases_mle),
        ];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));

        // --- Create input layers ---
        let live_committed_input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer_with_rho_inv(4_u8, 1_f64);

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // **************************** BEGIN: checking the bits are binary ****************************

        // --- Create the builders for (b_i)^2 - b_i ---
        let bits_are_binary_builder = BitsAreBinaryBuilder::new(self.mnist_input.relu_bin_decomp.clone());
        let bits_are_binary_result = layers.add_gkr(bits_are_binary_builder);

        // **************************** BEGIN: first matmul + bias circuit ****************************

        let l1_input_matrix = Matrix::new(
            self.mnist_input.input_mle.mle_ref(),
            0 as usize,
            log2(self.mnist_input.dim.num_features) as usize,
            self.mnist_input.input_mle.get_prefix_bits(),
        );

        let l1_weights_matrix = Matrix::new(
            self.mnist_weights.l1_linear_weights.weights_mle.mle_ref(),
            log2(self.mnist_weights.l1_linear_weights.dim.in_features) as usize,
            log2(self.mnist_weights.l1_linear_weights.dim.out_features) as usize,
            self.mnist_weights.l1_linear_weights.weights_mle.get_prefix_bits(),
        );

        let l1_matmul_out = layers.add_matmult_layer(l1_input_matrix, l1_weights_matrix);

        // --- Bias circuit ---
        let l1_bias_builder = BiasBuilder::new(l1_matmul_out, self.mnist_weights.l1_linear_weights.biases_mle.clone());
        let l1_matmul_bias_out = layers.add_gkr(l1_bias_builder);

        // **************************** BEGIN: checking the bits recompute to the mles ****************************

        // --- First we create the positive binary recomp builder ---
        let pos_bin_recomp_builder = PositiveBinaryRecompBuilder::new(self.mnist_input.relu_bin_decomp.clone());
        let pos_bin_recomp_mle = layers.add_gkr(pos_bin_recomp_builder);

        // --- Finally, the recomp checker ---
        let recomp_checker_builder = BinaryRecompCheckerBuilder::new(
            self.mnist_input.input_mle.clone(),
            self.mnist_input.relu_bin_decomp.clone(),
            pos_bin_recomp_mle,
        );
        let recomp_checker_result = layers.add_gkr(recomp_checker_builder);
        
        // **************************** BEGIN: the actual relu circuit ****************************

        // --- Create the builders for (1 - b_i) * x_i ---
        // ---   this simplifies to x_i - b_i * x_i    ---
        let l1_relu_builder = ReLUBuilder::new(self.mnist_input.relu_bin_decomp.clone(), l1_matmul_bias_out);
        let l1_relu_result = layers.add_gkr(l1_relu_builder);

        // **************************** BEGIN: the second linear circuit ****************************

        let l2_input_matrix = Matrix::new(
            l1_relu_result.mle_ref(),
            0 as usize,
            l1_relu_result.num_iterated_vars() as usize,
            l1_relu_result.get_prefix_bits(),
        );

        let l2_weights_matrix = Matrix::new(
            self.mnist_weights.l2_linear_weights.weights_mle.mle_ref(),
            log2(self.mnist_weights.l2_linear_weights.dim.in_features) as usize,
            log2(self.mnist_weights.l2_linear_weights.dim.out_features) as usize,
            self.mnist_weights.l2_linear_weights.weights_mle.get_prefix_bits(),
        );

        let l2_matmul_out = layers.add_matmult_layer(l2_input_matrix, l2_weights_matrix);

        // --- Bias circuit ---
        let l2_bias_builder = BiasBuilder::new(l2_matmul_out, self.mnist_weights.l2_linear_weights.biases_mle.clone());
        let l2_matmul_bias_out = layers.add_gkr(l2_bias_builder);

        Witness {
            layers,
            output_layers: vec![
                recomp_checker_result.get_enum(),
                bits_are_binary_result.get_enum(),
                l2_matmul_bias_out.mle_ref().get_enum(),
            ],
            input_layers: vec![live_committed_input_layer.to_enum()]
        }
    }
}


impl<F: FieldExt> MNISTModelCircuit<F> {
    /// Creates a new instance of BinDecomp16BitsAreBinaryCircuit
    pub fn new(
        mnist_weights: MNISTWeights<F>,
        mnist_input: MNISTInputData<F>,
    ) -> Self {
        Self {
            mnist_weights,
            mnist_input,
        }
    }
}
#[ignore]
#[test]
fn test_relu_circuit() {


    let l1_dim = NNLinearDimension {
        in_features: 784,
        out_features: 200,
    };
    let l2_dim = NNLinearDimension {
        in_features: 200,
        out_features: 10,
    };
    let input_dim = NNLinearInputDimension {
        num_features: 784,
    };
    let (mnist_weights, mnist_inputs) = load_dummy_mnist_input_and_weights(
        l1_dim,
        l2_dim,
        input_dim,
    );

    let mut circuit = MNISTModelCircuit::new(
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