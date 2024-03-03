use std::path::Path;

use ark_std::{log2, test_rng};
use rand::Rng;
use remainder::{layer::{matmult::Matrix, LayerId}, mle::{dense::DenseMle, Mle, MleRef}, prover::{helpers::test_circuit, input_layer::{combine_input_layers::InputLayerBuilder, enum_input_layer::InputLayerEnum, ligero_input_layer::LigeroInputLayer, InputLayer}, GKRCircuit, GKRError, Layers, Witness}};
use remainder_shared_types::{transcript::poseidon_transcript::PoseidonTranscript, FieldExt, Fr};



pub struct MatMulCircuit<F: FieldExt> {
    pub input_mle: DenseMle<F, F>,      // represent matrix on the left, shape (sample_size, in_features)
    pub weights_mle: DenseMle<F, F>,    // represent matrix on the right, note this is the A^T matrix from: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
                                        // ^ its shape is (in_features, out_features)
    pub sample_size: usize,
    pub in_features: usize,
    pub out_features: usize,
}

impl<F: FieldExt> GKRCircuit<F> for MatMulCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        
        let input_mles_commit: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut self.input_mle),
            Box::new(&mut self.weights_mle),
        ];

        let input_mles_commit_builder =
            InputLayerBuilder::new(input_mles_commit, None, LayerId::Input(0));

        let commit_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> =
            input_mles_commit_builder.to_input_layer_with_rho_inv(
                4,
                64.
            );

        let commit_input_layer = commit_input_layer.to_enum();

        let input_layers = vec![commit_input_layer.to_enum()];

        let mut layers = Layers::new();

        let input_matrix = Matrix::new(
            self.input_mle.mle_ref(),
            self.sample_size as usize,
            self.in_features as usize,
            self.input_mle.get_prefix_bits(),
        );

        let weights_matrix = Matrix::new(
            self.weights_mle.mle_ref(),
            self.in_features as usize,
            self.out_features as usize,
            self.weights_mle.get_prefix_bits(),
        );

        let matmult_out = layers.add_matmult_layer(input_matrix, weights_matrix);

        let output_layers = vec![matmult_out.mle_ref().get_enum()];

        Witness {
            layers,
            output_layers,
            input_layers,
        }
    }
}


#[test]
fn test_matmul_circuit() {
    let mut rng = test_rng();

    let input_mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..1 << 7).map(|_| Fr::from(rng.gen::<u64>()).into()),
        LayerId::Input(0),
        None,
    );  // input_mle's shape is (16, 8) -> (sample_size, in_features)

    let weights_mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..1 << 5).map(|_| Fr::from(rng.gen::<u64>()).into()),
        LayerId::Input(0),
        None,
    );  // weights_mle's shape is (8, 4) -> (in_features, out_features)

    let circuit: MatMulCircuit<Fr> = MatMulCircuit {
        input_mle,
        weights_mle,
        sample_size: 16,
        in_features: 8,
        out_features: 4,
    };
    
    test_circuit(circuit, Some(Path::new("matmul_circuit.json")));
}