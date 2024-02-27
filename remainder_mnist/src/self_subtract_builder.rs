use remainder::{expression::ExpressionStandard, layer::LayerBuilder, mle::{dense::DenseMle, structs::BinDecomp16Bit, zero::ZeroMleRef, Mle}};
use remainder_shared_types::FieldExt;

pub struct SelfSubtractBuilder<F: FieldExt> {
    self_subtract_mle: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for SelfSubtractBuilder<F> {
    type Successor = ZeroMleRef<F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Mle(self.self_subtract_mle.mle_ref()) - ExpressionStandard::Mle(self.self_subtract_mle.mle_ref())
    }

    fn next_layer(&self, id: remainder::layer::LayerId, prefix_bits: Option<Vec<remainder::mle::MleIndex<F>>>) -> Self::Successor {
        ZeroMleRef::new(self.self_subtract_mle.num_iterated_vars(), prefix_bits, id)
    }
}

impl<F: FieldExt> SelfSubtractBuilder<F> {
    pub fn new(
        self_subtract_mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            self_subtract_mle: self_subtract_mle
        }
    }
}