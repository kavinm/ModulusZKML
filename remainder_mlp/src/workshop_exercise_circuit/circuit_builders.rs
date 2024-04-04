// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
