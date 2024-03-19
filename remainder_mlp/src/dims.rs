use remainder::mle::{bin_decomp_structs::bin_decomp_64_bit::BinDecomp64Bit, dense::DenseMle};
use remainder_shared_types::FieldExt;

#[derive(Clone)]
pub struct NNLinearDimension {
    pub in_features: usize,
    pub out_features: usize,
}

#[derive(Clone)]
pub struct NNLinearInputDimension {
    pub num_features: usize,
}

#[derive(Clone)]
pub struct NNLinearWeights<F: FieldExt> {
    pub weights_mle: DenseMle<F, F>,    // represent matrix on the right, note this is the A^T matrix from: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
                                        // ^ its shape is (in_features, out_features)
    pub biases_mle: DenseMle<F, F>,     // represent the biases, shape (out_features,)
    pub dim: NNLinearDimension,
}

#[derive(Clone)]
pub struct MLPWeights<F: FieldExt> {
    pub all_linear_weights_biases: Vec<NNLinearWeights<F>>,
}

/// Not batched
type ReluWitness<F> = DenseMle<F, BinDecomp64Bit<F>>;

#[derive(Clone)]
pub struct MLPInputData<F: FieldExt> {
    pub input_mle: DenseMle<F, F>,
    pub dim: NNLinearInputDimension,
    pub relu_bin_decomp: Vec<ReluWitness<F>>,
}

#[derive(Clone)]
pub struct DataparallelMLPInputData<F: FieldExt> {
    pub input_mles: Vec<DenseMle<F, F>>,
    pub dim: NNLinearInputDimension,
    pub relu_bin_decomp: Vec<Vec<ReluWitness<F>>>,
}