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
