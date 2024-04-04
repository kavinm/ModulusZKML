// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use remainder::{
    expression::ExpressionStandard,
    layer::LayerBuilder,
    mle::{
        dense::{DenseMle, DenseMleRef},
        zero::ZeroMleRef,
        Mle,
    },
};
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

    fn next_layer(
        &self,
        id: remainder::layer::LayerId,
        prefix_bits: Option<Vec<remainder::mle::MleIndex<F>>>,
    ) -> Self::Successor {
        let result_bookkeeping_table = self
            .data_mle
            .into_iter()
            .zip(self.bias_mle.into_iter())
            .map(|(data, bias)| data + bias);
        DenseMle::new_from_iter(result_bookkeeping_table, id, prefix_bits)
    }
}

impl<F: FieldExt> BiasBuilder<F> {
    pub fn new(data_mle: DenseMle<F, F>, bias_mle: DenseMle<F, F>) -> Self {
        Self { data_mle, bias_mle }
    }
}
