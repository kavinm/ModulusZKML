// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use remainder::mle::{bin_decomp_structs::bin_decomp_32_bit::BinDecomp32Bit, dense::DenseMle};
use remainder_shared_types::FieldExt;

#[derive(Clone, Debug)]
pub struct NNLinearDimension {
    pub in_features: usize,
    pub out_features: usize,
}

#[derive(Clone, Debug)]
pub struct NNLinearInputDimension {
    pub num_features: usize,
}

#[derive(Clone, Debug)]
pub struct DataparallelNNLinearInputDimension {
    pub batch_size: usize,
    pub num_features: usize,
}

#[derive(Clone, Debug)]
pub struct NNLinearWeights<F: FieldExt> {
    pub weights_mle: DenseMle<F, F>, // represent matrix on the right, note this is the A^T matrix from: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    // ^ its shape is (in_features, out_features)
    pub biases_mle: DenseMle<F, F>, // represent the biases, shape (out_features,)
    pub dim: NNLinearDimension,
}

#[derive(Clone, Debug)]
pub struct MLPWeights<F: FieldExt> {
    pub hidden_linear_weight_bias: NNLinearWeights<F>,
    pub out_linear_weight_bias: NNLinearWeights<F>,
}

/// Not batched
type ReluWitness<F> = DenseMle<F, BinDecomp32Bit<F>>;

#[derive(Clone, Debug)]
pub struct MLPInputData<F: FieldExt> {
    pub input_mle: DenseMle<F, F>,
    pub dim: NNLinearInputDimension,
    pub relu_bin_decomp: ReluWitness<F>,
}
