// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use remainder::{
    expression::ExpressionStandard,
    layer::LayerBuilder,
    mle::{bin_decomp_structs::bin_decomp_32_bit::BinDecomp32Bit, dense::DenseMle},
};
use remainder_shared_types::FieldExt;

/// Builder which computes the ReLU(x) function using the original value
/// `x` and the binary decomposition {b_0, ..., b_{30}, b_s} of `x`.
///
/// Recall that ReLU(x) = max(0, x)
pub struct ReLUBuilder<F: FieldExt> {
    signed_bin_decomp_mle: DenseMle<F, BinDecomp32Bit<F>>,
    orig_mle: DenseMle<F, F>,
}

/// The LayerBuilder trait requires you to implement two functions:
/// the build expression function and the next layer function
impl<F: FieldExt> LayerBuilder<F> for ReLUBuilder<F> {
    /// The `Self::Successor` type defines the output type of the `next_layer`
    /// function, i.e. the type of MleRef which this layer computes from its
    /// input MLEs.
    type Successor = DenseMle<F, F>;

    /// The `build_expression` function returns an expression representing the
    /// polynomial relationship between the input MLEs (i.e. those present
    /// within the `struct ReLUBuilder`) and the output of the
    /// `ReLUBuilder` circuit layer.
    fn build_expression(&self) -> ExpressionStandard<F> {
        todo!()
    }

    /// The `next_layer` function performs the actual "computation" of the
    /// circuit, i.e. using the values within `self.signed_bin_decomp_mle` and
    /// `self.orig_mle`, it should output a `Self::Successor` whose values
    /// represent ReLU(self.orig_mle).
    fn next_layer(
        &self,
        id: remainder::layer::LayerId,
        prefix_bits: Option<Vec<remainder::mle::MleIndex<F>>>,
    ) -> Self::Successor {
        todo!()
    }
}

impl<F: FieldExt> ReLUBuilder<F> {
    /// constructor for our bits are binary layer builder!
    pub fn new(
        signed_bin_decomp_mle: DenseMle<F, BinDecomp32Bit<F>>,
        orig_mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            signed_bin_decomp_mle,
            orig_mle,
        }
    }
}
