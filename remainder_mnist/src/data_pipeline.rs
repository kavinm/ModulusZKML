use remainder::mle::{dense::DenseMle, structs::BinDecomp16Bit};
use remainder_shared_types::FieldExt;

pub struct NNLinearDimension {
    pub in_features: usize,
    pub out_features: usize,
}

pub struct NNLinearWeights<F: FieldExt> {
    pub weights_mle: DenseMle<F, F>,    // represent matrix on the right, note this is the A^T matrix from: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
                                        // ^ its shape is (in_features, out_features)
    pub biases_mle: DenseMle<F, F>,     // represent the biases, shape (out_features,)
    pub dim: NNLinearDimension,
}

pub struct MNISTWeights<F: FieldExt> {
    pub l1_linear_weights: NNLinearWeights<F>,
    pub l2_linear_weights: NNLinearWeights<F>,
}

type ReluWitness<F> = Vec<DenseMle<F, BinDecomp16Bit<F>>>;

pub struct MNISTInputData<F: FieldExt> {
    pub input_mle: DenseMle<F, F>,      // represent the input matrix has shape (sample_size, features)
    pub sample_size: usize,
    pub relu_bin_decomp: ReluWitness<F>,
}