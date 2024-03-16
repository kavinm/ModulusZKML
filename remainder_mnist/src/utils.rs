use ark_std::{log2, test_rng};
use itertools::Itertools;
use rand::Rng;
use rayon::str;
use remainder::{layer::{matmult::{product_two_matrices, Matrix}, LayerId}, mle::{dense::DenseMle, structs::BinDecomp16Bit, Mle}};
use remainder_shared_types::{halo2curves::FieldExt as Halo2FieldExt, FieldExt, Fr};

use crate::data_pipeline::{MNISTInputData, MNISTWeights, NNLinearDimension, NNLinearInputDimension, NNLinearWeights};


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
    in_features: usize,
) -> (
    DenseMle<F, BinDecomp16Bit<F>>,
    DenseMle<F, F>,
) {
    let mut rng = test_rng();

    let bin_decomp_16_bits: Vec<[F; 16]> = 
        (0..in_features).map(
            |_| (0..16).map(
                |_| F::from(rng.gen_range(0..=1))
            ).collect_vec().try_into().unwrap()
        ).collect();

    let mle_bin_decomp_16_bits = 
        DenseMle::new_from_iter(
            bin_decomp_16_bits.clone().into_iter().map(
                |in_feature| BinDecomp16Bit {
                    bits: in_feature,
                }
            ), LayerId::Input(0), None
    );

    let bin_decomp_recomp: Vec<F> = 
        bin_decomp_16_bits.clone().iter().map(
            |in_feature| recompute_16_bit_decomp_signed(in_feature)
        ).collect();

    let mle_bin_decomp_recomp = 
        DenseMle::new_from_iter(
                bin_decomp_recomp.into_iter(),
                LayerId::Input(0),
                None,
        );

    (mle_bin_decomp_16_bits, mle_bin_decomp_recomp)

}

pub fn gen_random_nn_linear_weights<F: FieldExt>(
    dim: NNLinearDimension,
) -> NNLinearWeights<F> {
    let mut rng = test_rng();

    let weights_mle: DenseMle<F, F> = DenseMle::new_from_iter(
        (0.. (dim.in_features * dim.out_features)).map(|_| F::from(rng.gen_range(0..=15)).into()),
        LayerId::Input(0),
        None,
    );

    let biases_mle: DenseMle<F, F> = DenseMle::new_from_iter(
        (0..dim.out_features).map(|_| F::from(rng.gen_range(0..=15)).into()),
        LayerId::Input(0),
        None,
    );

    NNLinearWeights {
        weights_mle,
        biases_mle,
        dim,
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

/// Return the `bit_length` bit signed decomposition of the specified i32, or None if the argument is too large.
/// Result is little endian (so LSB has index 0).
/// Sign bit has maximal index.
/// Pre: bit_length > 1.
pub fn build_signed_bit_decomposition(value: i32, bit_length: usize) -> Option<Vec<bool>> {
    let unsigned = build_unsigned_bit_decomposition(value.unsigned_abs(), bit_length - 1);
    if let Some(mut bits) = unsigned {
        bits.push(value < 0);
        return Some(bits);
    }
    None
}

/// Return the `bit_length` bit decomposition of the specified u32, or None if the argument is too large.
/// Result is little endian i.e. LSB has index 0.
pub fn build_unsigned_bit_decomposition(mut value: u32, bit_length: usize) -> Option<Vec<bool>> {
    let mut bits = vec![];
    for _ in 0..bit_length {
        bits.push((value & 1) != 0);
        value >>= 1;
    }
    if value == 0 {
        Some(bits)
    } else {
        None
    }
}

pub fn load_dummy_mnist_input_data(
    l1_weights: &NNLinearWeights<Fr>,
    input_dim: NNLinearInputDimension,
) -> MNISTInputData<Fr> {

    let mut rng = test_rng();

    let input_data: Vec<u64> = (0..(input_dim.num_features)).map(|_| rng.gen_range(0..=15)).collect_vec();

    let input_mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        input_data.into_iter().map(|x| Fr::from(x)),
        LayerId::Input(0),
        None,
    );

    dbg!("Okay got here! 1");
    dbg!(input_mle.num_iterated_vars());
    let input_matrix = Matrix::new(
        input_mle.mle_ref(),
        1 as usize,
        input_dim.num_features,
        input_mle.prefix_bits.clone(),
    );
    dbg!("Okay got here! 2");

    let weights_matrix = Matrix::new(
        l1_weights.weights_mle.mle_ref(),
        l1_weights.dim.in_features,
        l1_weights.dim.out_features,
        l1_weights.weights_mle.prefix_bits.clone(),
    );

    // println!("INPUT MATRIX {:?}", input_matrix.num_rows_vars);
    // println!("INPUT MATRIX {:?}", input_matrix.num_cols_vars);
    // println!("INPUT MATRIX {:?}", input_mle.mle_ref().bookkeeping_table.len());
    // println!("weights_matrix MATRIX {:?}", weights_matrix.num_rows_vars);
    // println!("weights_matrix MATRIX {:?}", weights_matrix.num_cols_vars);
    // println!("weights_matrix MATRIX {:?}", l1_weights.weights_mle.mle_ref().bookkeeping_table.len());

    let l1_out = product_two_matrices(input_matrix, weights_matrix);
    let l1_out_w_bias = l1_out.iter().zip(
        l1_weights.biases_mle.mle.iter()
    ).map(|(x, bias)| {
        x + bias
    }).collect_vec();

    // --- Need to compute binary decomp of the result of the first matmul ---
    let relu_decomp = l1_out_w_bias.into_iter().map(|x| {
        let field_elem_u32_repr = x.get_lower_128() as u32;
        let decomp = build_unsigned_bit_decomposition(field_elem_u32_repr, 16).unwrap();
        BinDecomp16Bit::<Fr>::from(decomp)
    }).collect_vec();

    // let l1_out_w_bias_w_relu = l1_out_w_bias.into_iter().map(|x| {
    //     if x > Fr::zero() {x} else {Fr::zero()}
    // });

    MNISTInputData {
        input_mle,
        dim: input_dim,
        relu_bin_decomp: DenseMle::new_from_iter(
            relu_decomp.into_iter(),
            LayerId::Input(0),
            None,
        ),
    }

}

pub fn load_dummy_mnist_input_and_weights(
    l1_dim: NNLinearDimension,
    l2_dim: NNLinearDimension,
    input_dim: NNLinearInputDimension,
) -> (
    MNISTWeights<Fr>,
    MNISTInputData<Fr>,
) {
    let mnist_weights = load_dummy_mnist_model_weights(l1_dim, l2_dim);
    let mnist_input_data = load_dummy_mnist_input_data(&mnist_weights.l1_linear_weights, input_dim);
    (mnist_weights, mnist_input_data)
}


/// Loads the actual MNIST model weights from the given file path.
pub fn load_mnist_model_weights() -> () {
    todo!()
}

/// Loads the actual MNIST model inputs from the given file path.
pub fn load_mnist_input_data() -> () {
    todo!()
}