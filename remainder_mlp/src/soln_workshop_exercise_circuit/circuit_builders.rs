use remainder::layer::LayerId;
use remainder::mle::bin_decomp_structs::bin_decomp_32_bit::BinDecomp32Bit;
use remainder::mle::MleIndex;
use remainder::{
    expression::ExpressionStandard,
    layer::LayerBuilder,
    mle::{dense::DenseMle, zero::ZeroMleRef, Mle},
};
use remainder_shared_types::FieldExt;

/// This builder computes the value `pos_recomp` - `x` + 2 * `sign_bit` * `x`.
/// Note that this is equivalent to
/// (1 - b_s)(`pos_recomp` - `x`) + `b_s`(`pos_recomp` + `x`)
///
/// This checks that the sign bit and `pos_recomp` are as they
/// purport to be, assuming a range check on `pos_recomp`.
pub struct BinaryRecompCheckerBuilder<F: FieldExt> {
    mle: DenseMle<F, F>,
    signed_bit_decomp_mle: DenseMle<F, BinDecomp32Bit<F>>,
    pos_recomp_mle: DenseMle<F, F>,
}
impl<F: FieldExt> LayerBuilder<F> for BinaryRecompCheckerBuilder<F> {
    type Successor = ZeroMleRef<F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        // --- Grab MLE refs ---
        let signed_bit_mle_ref = self.signed_bit_decomp_mle.mle_bit_refs()
            [self.signed_bit_decomp_mle.mle_bit_refs().len() - 1]
            .clone();
        let value_mle_ref = self.mle.mle_ref();

        // --- LHS of addition ---
        let pos_recomp_minus_diff = ExpressionStandard::Mle(self.pos_recomp_mle.mle_ref())
            - ExpressionStandard::Mle(value_mle_ref.clone());

        // --- RHS of addition ---
        let sign_bit_times_diff_ptr = Box::new(ExpressionStandard::Product(vec![
            signed_bit_mle_ref,
            value_mle_ref,
        ]));
        let two_times_sign_bit_times_diff =
            ExpressionStandard::Scaled(sign_bit_times_diff_ptr, F::from(2));

        pos_recomp_minus_diff + two_times_sign_bit_times_diff
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        ZeroMleRef::new(self.mle.num_iterated_vars(), prefix_bits, id)
    }
}

impl<F: FieldExt> BinaryRecompCheckerBuilder<F> {
    /// Constructor
    pub fn new(
        mle: DenseMle<F, F>,
        signed_bit_decomp_mle: DenseMle<F, BinDecomp32Bit<F>>,
        pos_recomp_mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            mle,
            signed_bit_decomp_mle,
            pos_recomp_mle,
        }
    }
}
