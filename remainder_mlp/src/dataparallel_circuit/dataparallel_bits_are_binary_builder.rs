use remainder::{expression::ExpressionStandard, layer::LayerBuilder, mle::{bin_decomp_structs::bin_decomp_64_bit::BinDecomp64Bit, dense::DenseMle, structs::BinDecomp16Bit, zero::ZeroMleRef, Mle}};
use remainder_shared_types::FieldExt;

pub struct DataparallelBitsAreBinaryBuilder<F: FieldExt> {
    signed_bin_decomp_mle: DenseMle<F, BinDecomp64Bit<F>>,
}

impl<F: FieldExt> LayerBuilder<F> for DataparallelBitsAreBinaryBuilder<F> {
    type Successor = ZeroMleRef<F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        let signed_bit_mle_ref = self.signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
        ExpressionStandard::Mle(signed_bit_mle_ref.clone()) - ExpressionStandard::Product(vec![signed_bit_mle_ref.clone(), signed_bit_mle_ref.clone()])
    }

    fn next_layer(&self, id: remainder::layer::LayerId, prefix_bits: Option<Vec<remainder::mle::MleIndex<F>>>) -> Self::Successor {
        ZeroMleRef::new(self.signed_bin_decomp_mle.num_iterated_vars(), prefix_bits, id)
    }
}

impl<F: FieldExt> DataparallelBitsAreBinaryBuilder<F> {
    /// constructor for our bits are binary layer builder!
    pub fn new(
        signed_bin_decomp_mle: DenseMle<F, BinDecomp64Bit<F>>,
    ) -> Self {
        Self {
            signed_bin_decomp_mle
        }
    }
}