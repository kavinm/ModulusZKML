use ark_std::test_rng;
use itertools::Itertools;
use rand::Rng;
use rayon::str;
use remainder::{layer::LayerId, mle::{dense::DenseMle, structs::BinDecomp16Bit}};
use remainder_shared_types::{FieldExt, Fr};


pub fn recompute_16_bit_decomp<F: FieldExt>(
    decomp_bits: &[F; 16],
) -> F {
    // skip 1 because the last bit is the signed bit
    decomp_bits.iter().rev().enumerate().fold(
        F::zero(), |acc, (bit_idx, bit)| {
        acc + *bit * F::from(2_u64.pow((16 - (bit_idx + 1)) as u32))
    })
}

pub fn generate_16_bit_decomp<F: FieldExt>(
    sample_size: usize,
    in_features: usize,
) -> (
    Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    Vec<DenseMle<F, F>>,
) {
    let mut rng = test_rng();

    let bin_decomp_16_bits: Vec<Vec<[F; 16]>> = (0..sample_size).map(
        |_| (0..in_features).map(
            |_| (0..16).map(
                |_| F::from(rng.gen_range(0..=1))
            ).collect_vec().try_into().unwrap()
        ).collect()
    ).collect();

    let mle_bin_decomp_16_bits = bin_decomp_16_bits.clone().into_iter().map(
        |sample| DenseMle::new_from_iter(
                sample.into_iter().map(
                    |in_feature| BinDecomp16Bit {
                        bits: in_feature,
                    }
                ), LayerId::Input(0), None
        )
    ).collect_vec();

    let bin_decomp_recomp: Vec<Vec<F>> = bin_decomp_16_bits.iter().map(
        |sample| sample.iter().map(
            |in_feature| recompute_16_bit_decomp(in_feature)
        ).collect()
    ).collect();

    let mle_bin_decomp_recomp = bin_decomp_recomp.into_iter().map(
        |sample| DenseMle::new_from_iter(
                sample.into_iter(),
                LayerId::Input(0),
                None,
        )
    ).collect();

    (mle_bin_decomp_16_bits, mle_bin_decomp_recomp)

}


pub fn recompute_16_bit_decomp_signed<F: FieldExt>(
    decomp_bits: &[F; 16],
) -> F {

    // skip 1 because the last bit is the signed bit
    let unsigned_val = decomp_bits.iter().rev().enumerate().skip(1).fold(
        F::zero(), |acc, (bit_idx, bit)| {
        acc + *bit * F::from(2_u64.pow((16 - (bit_idx + 1)) as u32))
    });

    // signed bit equals 1 -> negative
    if decomp_bits[15] == F::one() {
        -unsigned_val
    } else if decomp_bits[15] == F::zero() {
        unsigned_val
    } else {
        panic!("Invalid signed bit")
    }
}

pub fn generate_16_bit_decomp_signed<F: FieldExt>(
    sample_size: usize,
    in_features: usize,
) -> (
    Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    Vec<DenseMle<F, F>>,
) {
    let mut rng = test_rng();

    let bin_decomp_16_bits: Vec<Vec<[F; 16]>> = (0..sample_size).map(
        |_| (0..in_features).map(
            |_| (0..16).map(
                |_| F::from(rng.gen_range(0..=1))
            ).collect_vec().try_into().unwrap()
        ).collect()
    ).collect();

    let mle_bin_decomp_16_bits = bin_decomp_16_bits.clone().into_iter().map(
        |sample| DenseMle::new_from_iter(
                sample.into_iter().map(
                    |in_feature| BinDecomp16Bit {
                        bits: in_feature,
                    }
                ), LayerId::Input(0), None
        )
    ).collect_vec();

    let bin_decomp_recomp: Vec<Vec<F>> = bin_decomp_16_bits.iter().map(
        |sample| sample.iter().map(
            |in_feature| recompute_16_bit_decomp_signed(in_feature)
        ).collect()
    ).collect();

    let mle_bin_decomp_recomp = bin_decomp_recomp.into_iter().map(
        |sample| DenseMle::new_from_iter(
                sample.into_iter(),
                LayerId::Input(0),
                None,
        )
    ).collect();

    (mle_bin_decomp_16_bits, mle_bin_decomp_recomp)

}

pub struct NNLinearWeights<F: FieldExt> {
    pub weights_mle: DenseMle<F, F>,    // represent matrix on the right, note this is the A^T matrix from: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
                                        // ^ its shape is (in_features, out_features)
    pub biases_mle: DenseMle<F, F>,     // represent the biases, shape (out_features,)
    pub in_features: usize,
    pub out_features: usize,
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

pub struct NNLinearDimension {
    pub in_features: usize,
    pub out_features: usize,
}

pub fn gen_random_nn_linear_weights<F: FieldExt>(
    dim: NNLinearDimension,
) -> NNLinearWeights<F> {
    let mut rng = test_rng();

    let weights_mle: DenseMle<F, F> = DenseMle::new_from_iter(
        (0.. (dim.in_features * dim.out_features)).map(|_| F::from(rng.gen::<u64>()).into()),
        LayerId::Input(0),
        None,
    );

    let biases_mle: DenseMle<F, F> = DenseMle::new_from_iter(
        (0..1 << dim.out_features).map(|_| F::from(rng.gen::<u64>()).into()),
        LayerId::Input(0),
        None,
    );

    NNLinearWeights {
        weights_mle,
        biases_mle,
        in_features: dim.in_features,
        out_features: dim.out_features,
    }
}

pub fn load_dummy_mnist_model_weights(
    l1_dim: NNLinearDimension,
    l2_dim: NNLinearDimension,
) -> MNISTWeights<Fr> {

    MNISTWeights {
        l1_linear_weights: gen_random_nn_linear_weights(l1_dim),
        l2_linear_weights: gen_random_nn_linear_weights(l2_dim),
    }
    
}

pub fn load_dummy_mnist_input_data() -> () {
    todo!()
}

pub fn load_mnist_model_weights() -> () {
    todo!()
}

pub fn load_mnist_input_data() -> () {
    todo!()
}