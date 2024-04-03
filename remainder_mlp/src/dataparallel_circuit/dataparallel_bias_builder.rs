use remainder::{expression::ExpressionStandard, layer::LayerBuilder, mle::{dense::{DenseMle, DenseMleRef}, zero::ZeroMleRef, Mle}};
use remainder_shared_types::FieldExt;

pub struct BiasBuilder<F: FieldExt> {
    data_mle: DenseMle<F, F>,
    bias_mle: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for BiasBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        let data_mle_mle_ref = self.data_mle.mle_ref();
        let bias_mle_mle_ref = self.bias_mle.mle_ref();
        ExpressionStandard::Mle(data_mle_mle_ref) + ExpressionStandard::Mle(bias_mle_mle_ref)
    }

    fn next_layer(&self, id: remainder::layer::LayerId, prefix_bits: Option<Vec<remainder::mle::MleIndex<F>>>) -> Self::Successor {
        let result_bookkeeping_table = self.data_mle.into_iter().zip(self.bias_mle.into_iter().cycle()).map(|(data, bias)| {
            data + bias
        });
        DenseMle::new_from_iter(result_bookkeeping_table, id, prefix_bits)
    }
}

impl<F: FieldExt> BiasBuilder<F> {
    pub fn new(
        data_mle: DenseMle<F, F>,
        bias_mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            data_mle,
            bias_mle,
        }
    }
}