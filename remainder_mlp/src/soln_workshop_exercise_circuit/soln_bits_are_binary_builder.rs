// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use remainder::{
    expression::ExpressionStandard,
    layer::LayerBuilder,
    mle::{
        bin_decomp_structs::bin_decomp_32_bit::BinDecomp32Bit, dense::DenseMle, zero::ZeroMleRef,
        Mle,
    },
};
use remainder_shared_types::FieldExt;

pub struct BitsAreBinaryBuilder<F: FieldExt> {
    signed_bin_decomp_mle: DenseMle<F, BinDecomp32Bit<F>>,
}

impl<F: FieldExt> LayerBuilder<F> for BitsAreBinaryBuilder<F> {
    type Successor = ZeroMleRef<F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        let signed_bit_mle_ref = self.signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
        ExpressionStandard::Mle(signed_bit_mle_ref.clone())
            - ExpressionStandard::Product(vec![
                signed_bit_mle_ref.clone(),
                signed_bit_mle_ref.clone(),
            ])
    }

    fn next_layer(
        &self,
        id: remainder::layer::LayerId,
        prefix_bits: Option<Vec<remainder::mle::MleIndex<F>>>,
    ) -> Self::Successor {
        ZeroMleRef::new(
            self.signed_bin_decomp_mle.num_iterated_vars(),
            prefix_bits,
            id,
        )
    }
}

impl<F: FieldExt> BitsAreBinaryBuilder<F> {
    /// constructor for our bits are binary layer builder!
    pub fn new(signed_bin_decomp_mle: DenseMle<F, BinDecomp32Bit<F>>) -> Self {
        Self {
            signed_bin_decomp_mle,
        }
    }
}
