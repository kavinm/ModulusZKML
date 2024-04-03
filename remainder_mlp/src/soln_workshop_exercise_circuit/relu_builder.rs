use remainder::{
    expression::ExpressionStandard,
    layer::LayerBuilder,
    mle::{bin_decomp_structs::bin_decomp_32_bit::BinDecomp32Bit, dense::DenseMle},
};
use remainder_shared_types::FieldExt;

pub struct ReLUBuilder<F: FieldExt> {
    signed_bin_decomp_mle: DenseMle<F, BinDecomp32Bit<F>>,
    pos_recomp: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for ReLUBuilder<F> {
    type Successor = DenseMle<F, F>;

    /// (1 - b_s) * pos_recomp
    fn build_expression(&self) -> ExpressionStandard<F> {
        let signed_bit_mle_ref = self.signed_bin_decomp_mle.mle_bit_refs()
            [self.signed_bin_decomp_mle.mle_bit_refs().len() - 1]
            .clone();
        ExpressionStandard::Mle(self.pos_recomp.mle_ref())
            - ExpressionStandard::Product(vec![signed_bit_mle_ref, self.pos_recomp.mle_ref()])
    }

    fn next_layer(
        &self,
        id: remainder::layer::LayerId,
        prefix_bits: Option<Vec<remainder::mle::MleIndex<F>>>,
    ) -> Self::Successor {
        let result_iter = self.signed_bin_decomp_mle.mle_bit_refs()
            [self.signed_bin_decomp_mle.mle_bit_refs().len() - 1]
            .clone()
            .bookkeeping_table
            .into_iter()
            .zip(self.pos_recomp.into_iter())
            .map(|(signed_bit, mle)| (F::from(1) - signed_bit) * mle);
        DenseMle::new_from_iter(result_iter, id, prefix_bits)
    }
}

impl<F: FieldExt> ReLUBuilder<F> {
    /// constructor for our bits are binary layer builder!
    pub fn new(
        signed_bin_decomp_mle: DenseMle<F, BinDecomp32Bit<F>>,
        pos_recomp: DenseMle<F, F>,
    ) -> Self {
        Self {
            signed_bin_decomp_mle,
            pos_recomp,
        }
    }
}
