use remainder::{
    layer::LayerId,
    mle::{dense::DenseMle, Mle, MleRef},
    prover::{
        input_layer::{
            combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer,
            InputLayer,
        },
        GKRCircuit, Layers, Witness,
    },
};
use remainder_shared_types::{transcript::poseidon_transcript::PoseidonTranscript, FieldExt};

use super::circuit_builders::{DoubleAndSubtractBuilder, MleSquaringBuilder};

/// This circuit takes in as input V(b_1, ..., b_n) and W(b_1, ..., b_n),
/// and checks for all b_1, ..., b_n \in \{0, 1\}^n that
/// W(b_1, ..., b_n) = 2 * V(b_1, ..., b_n)^2.
pub struct SimpleArithmeticCircuit<F: FieldExt> {
    // --- These are the MLEs which the circuit has access to ---
    mle: DenseMle<F, F>,
    two_times_mle_squared: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for SimpleArithmeticCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // over here we construct the input layer to the circuit. You can basically see this as \tilde{V}_d(b_1, ...)
        // this is also done for you in the signed recomposition circuit!
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut self.mle),
            Box::new(&mut self.two_times_mle_squared),
        ];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let live_committed_input_layer: PublicInputLayer<F, Self::Transcript> =
            input_layer_builder.to_input_layer();

        // this is where we add the middle layers to the circuit.
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // This circuit has two separate layers. The first one computes
        // V(b_1, ..., b_n)^2, using the `MleSquaredBuilder` defined in
        // `circuit_builders.rs`.
        let mle_ref_to_be_squared = self.mle.mle_ref();
        let mle_squaring_builder = MleSquaringBuilder::new(mle_ref_to_be_squared);
        let mle_ref_squared = layers.add_gkr(mle_squaring_builder); // This is V(b_1, ..., b_n)^2, as defined by the `next_layer` function within `MleSquaringBuilder`.

        // The second one computes 2 * (V(b_1, ..., b_n)^2) - W(b_1, ..., b_n).
        let two_times_mle_squared_ref = self.two_times_mle_squared.mle_ref();
        let double_and_subtract_builder =
            DoubleAndSubtractBuilder::new(mle_ref_squared, two_times_mle_squared_ref);
        let output_zero_mle_ref = layers.add_gkr(double_and_subtract_builder);

        // return a witness that is then verified using our GKR verifier (see tests.rs)
        // The output of this circuit is simply the MLE represented by `output_zero_mle_ref`.
        Witness {
            layers,
            output_layers: vec![output_zero_mle_ref.get_enum()],
            input_layers: vec![live_committed_input_layer.to_enum()],
        }
    }
}

impl<F: FieldExt> SimpleArithmeticCircuit<F> {
    /// Creates a new instance of SimpleArithmeticCircuit
    pub fn new(mle: DenseMle<F, F>, two_times_mle_squared: DenseMle<F, F>) -> Self {
        Self {
            mle,
            two_times_mle_squared,
        }
    }
}
