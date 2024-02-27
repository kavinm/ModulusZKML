use remainder::{expression::ExpressionStandard, layer::LayerBuilder, mle::{dense::DenseMle, structs::BinDecomp16Bit, zero::ZeroMleRef, Mle}};
use remainder_shared_types::FieldExt;

pub struct BitsAreBinaryBuilder<F: FieldExt> {
    signed_bin_decomp_mle: DenseMle<F, BinDecomp16Bit<F>>,
}

impl<F: FieldExt> LayerBuilder<F> for BitsAreBinaryBuilder<F> {
    type Successor = ZeroMleRef<F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        let signed_bit_mle_ref = self.signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
        ExpressionStandard::Mle(signed_bit_mle_ref) - ExpressionStandard::Product(vec![signed_bit_mle_ref, signed_bit_mle_ref])
    }

    fn next_layer(&self, id: remainder::layer::LayerId, prefix_bits: Option<Vec<remainder::mle::MleIndex<F>>>) -> Self::Successor {
        ZeroMleRef::new(self.signed_bin_decomp_mle.num_iterated_vars(), prefix_bits, id)
    }
}

impl<F: FieldExt> BitsAreBinaryBuilder<F> {
    /// constructor for our bits are binary layer builder!
    pub fn new(
        signed_bin_decomp_mle: DenseMle<F, BinDecomp16Bit<F>>,
    ) -> Self {
        Self {
            signed_bin_decomp_mle
        }
    }
}