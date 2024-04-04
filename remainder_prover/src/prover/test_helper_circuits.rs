// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use remainder_shared_types::{transcript::Transcript, FieldExt};

use crate::{
    expression::ExpressionStandard,
    layer::{LayerBuilder, LayerId},
    mle::{dense::DenseMle, zero::ZeroMleRef, MleIndex},
};

/// Does the building for taking two MLEs of size two and multiplying each
/// to itself, then adding the results into a single MLE of size 1
pub struct EmptyLayerBuilder<F: FieldExt> {
    empty_layer_src_mle: DenseMle<F, F>,
    other_empty_layer_src_mle: DenseMle<F, F>,
}
impl<F: FieldExt> LayerBuilder<F> for EmptyLayerBuilder<F> {
    type Successor = DenseMle<F, F>;

    // --- Multiply `empty_layer_src_mle`'s elements and `other_empty_layer_src_mle`'s elements and add them together ---
    fn build_expression(&self) -> ExpressionStandard<F> {
        let split_mle_1 = self.empty_layer_src_mle.split(F::zero());
        let lhs = ExpressionStandard::products(vec![split_mle_1.first(), split_mle_1.second()]);

        let split_mle_2 = self.other_empty_layer_src_mle.split(F::zero());
        let rhs = ExpressionStandard::products(vec![split_mle_2.first(), split_mle_2.second()]);

        lhs + rhs
    }
    // --- Output should literally be a single element ---
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let result = (self.empty_layer_src_mle.mle[0] * self.empty_layer_src_mle.mle[1])
            + (self.other_empty_layer_src_mle.mle[0] * self.other_empty_layer_src_mle.mle[1]);

        DenseMle::new_from_raw(vec![result], id, prefix_bits)
    }
}
impl<F: FieldExt> EmptyLayerBuilder<F> {
    /// Constructor
    pub fn new(
        empty_layer_src_mle: DenseMle<F, F>,
        other_empty_layer_src_mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            empty_layer_src_mle,
            other_empty_layer_src_mle,
        }
    }
}

/// Subtracts the empty layer MLE value from the other passed-in MLE
pub struct EmptyLayerSubBuilder<F: FieldExt> {
    empty_layer_mle: DenseMle<F, F>,
    mle: DenseMle<F, F>,
}
impl<F: FieldExt> LayerBuilder<F> for EmptyLayerSubBuilder<F> {
    type Successor = ZeroMleRef<F>;

    // --- Literally just subtract them ---
    fn build_expression(&self) -> ExpressionStandard<F> {
        let empty_layer_mle_ref = ExpressionStandard::Mle(self.empty_layer_mle.mle_ref());
        let mle_mle_ref = ExpressionStandard::Mle(self.mle.mle_ref());
        mle_mle_ref - empty_layer_mle_ref
    }
    // --- Subtract them in a distributed fashion ---
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        // let result = self.mle.into_iter().map(|elem| {
        //     elem - self.empty_layer_mle.mle[0]
        // }).collect_vec();
        // DenseMle::<F, F>::new_from_raw(result, id, prefix_bits)

        ZeroMleRef::new(self.mle.mle_ref().num_vars, prefix_bits, id)
    }
}
impl<F: FieldExt> EmptyLayerSubBuilder<F> {
    /// Constructor
    pub fn new(empty_layer_mle: DenseMle<F, F>, mle: DenseMle<F, F>) -> Self {
        Self {
            empty_layer_mle,
            mle,
        }
    }
}

/// Adds the empty layer MLE value to the other passed-in MLE
pub struct EmptyLayerAddBuilder<F: FieldExt> {
    empty_layer_mle: DenseMle<F, F>,
    mle: DenseMle<F, F>,
}
impl<F: FieldExt> LayerBuilder<F> for EmptyLayerAddBuilder<F> {
    type Successor = ZeroMleRef<F>;

    // --- Literally just add them ---
    fn build_expression(&self) -> ExpressionStandard<F> {
        let empty_layer_mle_ref = ExpressionStandard::Mle(self.empty_layer_mle.mle_ref());
        let mle_mle_ref = ExpressionStandard::Mle(self.mle.mle_ref());
        mle_mle_ref + empty_layer_mle_ref
    }
    // --- Add them in a distributed fashion ---
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        // let result = self.mle.into_iter().map(|elem| {
        //     elem + self.empty_layer_mle.mle[0]
        // }).collect_vec();
        // DenseMle::<F, F>::new_from_raw(result, id, prefix_bits)

        ZeroMleRef::new(self.mle.mle_ref().num_vars, prefix_bits, id)
    }
}
impl<F: FieldExt> EmptyLayerAddBuilder<F> {
    /// Constructor
    pub fn new(empty_layer_mle: DenseMle<F, F>, mle: DenseMle<F, F>) -> Self {
        Self {
            empty_layer_mle,
            mle,
        }
    }
}
