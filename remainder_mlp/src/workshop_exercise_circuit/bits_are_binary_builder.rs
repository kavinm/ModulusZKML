use remainder::{
    expression::ExpressionStandard,
    layer::LayerBuilder,
    mle::{
        bin_decomp_structs::bin_decomp_32_bit::BinDecomp32Bit, dense::DenseMle, zero::ZeroMleRef,
        Mle,
    },
};
use remainder_shared_types::FieldExt;

/// Builder which should take in an MLE of purported binary decompositions
/// and ensure that all the contents of the MLE are actually in {0, 1}.
pub struct BitsAreBinaryBuilder<F: FieldExt> {
    bits_mle: DenseMle<F, BinDecomp32Bit<F>>,
}

impl<F: FieldExt> LayerBuilder<F> for BitsAreBinaryBuilder<F> {
    /// The `Self::Successor` type defines the output type of the `next_layer`
    /// function, i.e. the type of MleRef which this layer computes from its
    /// input MLEs. In this case, the check *should* result in all zeros!
    type Successor = ZeroMleRef<F>;

    /// The `build_expression` function returns an expression representing the
    /// polynomial relationship between the input MLEs (i.e. those present
    /// within the `struct BitsAreBinaryBuilder`) and the output of the
    /// `BitsAreBinaryBuilder` circuit layer.
    fn build_expression(&self) -> ExpressionStandard<F> {
        todo!()
    }

    fn next_layer(
        &self,
        id: remainder::layer::LayerId,
        prefix_bits: Option<Vec<remainder::mle::MleIndex<F>>>,
    ) -> Self::Successor {
        todo!()
    }
}

impl<F: FieldExt> BitsAreBinaryBuilder<F> {
    /// constructor for our bits are binary layer builder!
    pub fn new(bits_mle: DenseMle<F, BinDecomp32Bit<F>>) -> Self {
        Self { bits_mle }
    }
}
