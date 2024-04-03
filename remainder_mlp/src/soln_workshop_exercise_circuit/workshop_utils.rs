use ark_std::test_rng;
use itertools::Itertools;
use rand::Rng;
use remainder::{
    layer::{
        matmult::{product_two_matrices, Matrix},
        LayerId,
    },
    mle::{bin_decomp_structs::bin_decomp_32_bit::BinDecomp32Bit, dense::DenseMle, MleRef},
};
use remainder_shared_types::FieldExt;

use super::workshop_dims::{
    MLPInputData, MLPWeights, NNLinearDimension, NNLinearInputDimension, NNLinearWeights,
};

/// Primary function which we care about!
pub fn load_dummy_mlp_input_and_weights<F: FieldExt>(
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
) -> (MLPWeights<F>, MLPInputData<F>) {
    // --- Generate random inputs (8-bit) ---
    let input_mle = gen_random_mle(input_dim, 8);

    // --- Generate random weights/biases for both layers ---
    let hidden_linear_weight_bias: NNLinearWeights<F> = NNLinearWeights {
        weights_mle: gen_random_mle(input_dim * hidden_dim, 8),
        biases_mle: gen_random_mle(hidden_dim, 16),
        dim: NNLinearDimension {
            in_features: input_dim,
            out_features: hidden_dim,
        },
    };

    let out_linear_weight_bias: NNLinearWeights<F> = NNLinearWeights {
        weights_mle: gen_random_mle(hidden_dim * output_dim, 8),
        biases_mle: gen_random_mle(output_dim, 24),
        dim: NNLinearDimension {
            in_features: hidden_dim,
            out_features: output_dim,
        },
    };

    let mlp_weights_biases = MLPWeights {
        hidden_linear_weight_bias,
        out_linear_weight_bias,
    };

    // --- Generate bin decomps for all ReLU layers given inputs and weights ---
    let mlp_input_data = compute_hidden_layer_bin_decomp(&mlp_weights_biases, input_mle);

    (mlp_weights_biases, mlp_input_data)
}

/// Need to compute the binary decomp of the post-linear-layer outputs for
/// in-circuit hints.
pub fn compute_hidden_layer_bin_decomp<F: FieldExt>(
    mlp_weights_biases: &MLPWeights<F>,
    input_mle: DenseMle<F, F>,
) -> MLPInputData<F> {
    // --- Compute first linear layer ---
    let NNLinearWeights {
        weights_mle,
        biases_mle,
        dim,
    } = mlp_weights_biases.hidden_linear_weight_bias.clone();

    // --- Construct matrices for x^T A ---
    let input_matrix = Matrix::new(
        input_mle.mle_ref(),
        1 as usize,
        input_mle.mle_ref().bookkeeping_table().len(),
        input_mle.prefix_bits.clone(),
    );
    let weights_matrix = Matrix::new(
        weights_mle.mle_ref(),
        dim.in_features,
        dim.out_features,
        weights_mle.prefix_bits.clone(),
    );
    let affine_out_first_layer = product_two_matrices(input_matrix, weights_matrix);

    // println!("affine_out: {:?}", affine_out);

    // --- x^T A + b ---
    let linear_out_first_layer = affine_out_first_layer
        .into_iter()
        .zip(biases_mle.mle.iter())
        .map(|(x, bias)| x + bias)
        .collect_vec();

    // println!("linear_out: {:?}", linear_out);

    // --- Compute 32-bit bin decomp of `linear_out`, as required for ReLU in circuit ---
    let relu_decomp: Vec<BinDecomp32Bit<F>> = linear_out_first_layer
        .iter()
        .map(|x| {
            // println!("x: {:?}", x.get_lower_128());
            let mut field_elem_i64_repr = x.get_lower_128() as i64;
            if *x > F::from(1 << 63) {
                // println!("x: {:?}", x.get_lower_128());
                assert!(x.neg().get_lower_128() < (1 << 63));
                field_elem_i64_repr = -1 * (x.neg().get_lower_128() as i64);
                // maybe assert
            }
            // println!("field_elem_i64_repr: {:?}", field_elem_i64_repr);
            let decomp = build_signed_bit_decomposition(field_elem_i64_repr, 32).unwrap();
            BinDecomp32Bit::<F>::from(decomp)
        })
        .collect_vec();

    // println!("relu_decomp: {:?}", relu_decomp);

    let relu_decomp_mle =
        DenseMle::new_from_iter(relu_decomp.clone().into_iter(), LayerId::Input(0), None);

    MLPInputData {
        input_mle: input_mle.clone(),
        relu_bin_decomp: relu_decomp_mle,
        dim: NNLinearInputDimension {
            num_features: input_mle.mle_ref().bookkeeping_table().len(),
        },
    }
}

/// Grabs random MLE of fixed length and with values in the range
/// [-2^{total_bitwidth} - 1, 2^{total_bitwidth} - 1]
pub fn gen_random_mle<F: FieldExt>(length: usize, total_bitwidth: u32) -> DenseMle<F, F> {
    assert_ne!(total_bitwidth, 0);
    let mut rng = test_rng();
    // --- Generate random inputs ---
    let input_vec = (0..length).into_iter().map(|_| {
        let pos_in = F::from(rng.gen_range(0..=2_u64.pow(total_bitwidth - 1)));
        if rng.gen_bool(0.5) {
            return pos_in;
        } else {
            return pos_in.neg();
        }
    });

    DenseMle::new_from_iter(input_vec, LayerId::Input(0), None)
}

pub fn recompute_16_bit_decomp_signed<F: FieldExt>(decomp_bits: &[F; 16]) -> F {
    // skip 1 because the last bit is the signed bit
    let unsigned_val = decomp_bits
        .iter()
        .rev()
        .enumerate()
        .skip(1)
        .fold(F::zero(), |acc, (bit_idx, bit)| {
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

pub fn gen_random_nn_linear_weights<F: FieldExt>(dim: NNLinearDimension) -> NNLinearWeights<F> {
    let mut rng = test_rng();

    let weights_mle: DenseMle<F, F> = DenseMle::new_from_iter(
        (0..(dim.in_features * dim.out_features)).map(|_| F::from(rng.gen_range(0..=15)).into()),
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

/// Return the `bit_length` bit signed decomposition of the specified i64, or None if the argument is too large.
/// Result is little endian (so LSB has index 0).
/// Sign bit has maximal index.
/// Pre: bit_length > 1.
pub fn build_signed_bit_decomposition(value: i64, bit_length: usize) -> Option<Vec<bool>> {
    let unsigned = build_unsigned_bit_decomposition(value.unsigned_abs(), bit_length - 1);
    if let Some(mut bits) = unsigned {
        bits.push(value < 0);
        return Some(bits);
    }
    None
}

/// Return the `bit_length` bit decomposition of the specified u64, or None if the argument is too large.
/// Result is little endian i.e. LSB has index 0.
pub fn build_unsigned_bit_decomposition(mut value: u64, bit_length: usize) -> Option<Vec<bool>> {
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
