use ark_std::{log2, test_rng};
use itertools::Itertools;
use rand::Rng;
use remainder::{layer::{matmult::{product_two_matrices, Matrix}, LayerId}, mle::{bin_decomp_structs::bin_decomp_64_bit::BinDecomp64Bit, dense::DenseMle, Mle, MleRef}};
use remainder_shared_types::FieldExt;

use crate::dims::{DataparallelMLPInputData, DataparallelNNLinearInputDimension, MLPInputData, MLPWeights, NNLinearDimension, NNLinearInputDimension, NNLinearWeights};

/// Primary function which we care about!
pub fn load_batched_dummy_mlp_input_and_weights<F: FieldExt>(
    input_dim: usize,
    hidden_dims: Option<Vec<usize>>,
    output_dim: usize,
    batch_size: usize,
) -> (
    MLPWeights<F>,
    DataparallelMLPInputData<F>,
) {
    // --- Generate random inputs ---
    let input_mles = vec![gen_random_mle(input_dim, 16); batch_size];

    // --- Generate random weights/biases for all layers ---
    let all_input_dims = std::iter::once(input_dim)
        .chain(hidden_dims.clone().unwrap_or(vec![]));
    let all_output_dims = hidden_dims.unwrap_or(vec![])
        .into_iter()
        .chain(std::iter::once(output_dim));
    let all_linear_weights_biases = all_input_dims.zip(all_output_dims).map(|(input_dim, output_dim)| {
        NNLinearWeights {
            weights_mle: gen_random_mle(input_dim * output_dim, 16),
            biases_mle: gen_random_mle(output_dim, 16),
            dim: NNLinearDimension {
                in_features: input_dim,
                out_features: output_dim,
            },
        }
    }).collect_vec();

    let mlp_weights_biases = MLPWeights {
        all_linear_weights_biases,
    };

    // --- Generate bin decomps for all ReLU layers given inputs and weights ---
    let mlp_input_data = compute_batched_hidden_layer_values_and_bin_decomps(
        &mlp_weights_biases,
        input_mles,
    );

    (mlp_weights_biases, mlp_input_data)
}

/// We assume that after every linear layer there is a non-linearity,
/// and therefore compute both the literal values and the bin decomp
/// for those values for every hidden layer.
pub fn compute_batched_hidden_layer_values_and_bin_decomps<F: FieldExt>(
    mlp_weights_biases: &MLPWeights<F>,
    input_mles: Vec<DenseMle<F, F>>,
) -> DataparallelMLPInputData<F> {

    let batch_size = input_mles.len();
    let num_features = input_mles[0].mle_ref().bookkeeping_table().len();

    // --- Basically we are saving the intermediate outputs from BEFORE ReLU is computed ---
    let (hidden_values_decomp, final_val) = mlp_weights_biases.all_linear_weights_biases
        .clone()
        .into_iter()
        // --- No ReLU for last layer ---
        .rev()
        .skip(1)
        .rev()
        // ------------------------------
        .fold((vec![], input_mles.clone()), |(hidden_values_decomp_acc, last_layer_output), linear_weights_biases| {

            let NNLinearWeights { weights_mle, biases_mle, dim } = linear_weights_biases;

            let combined_last_layer_output = DenseMle::<F, F>::combine_mle_batch(last_layer_output.clone());

            // --- Construct matrices for x^T A ---
            let input_matrix = Matrix::new(
                combined_last_layer_output.mle_ref(),
                batch_size as usize,
                dim.in_features,
                combined_last_layer_output.prefix_bits.clone(),
            );
            let weights_matrix = Matrix::new(
                weights_mle.mle_ref(),
                dim.in_features,
                dim.out_features,
                weights_mle.prefix_bits.clone(),
            );
            let affine_out = product_two_matrices(input_matrix, weights_matrix);

            let affine_out_vec = affine_out.chunks(batch_size)
                .map(|chunk| chunk.to_vec())
                .collect_vec();

            assert_eq!(affine_out_vec.len(), batch_size);
            assert_eq!(affine_out_vec[0].len(), dim.out_features);

            // println!("affine_out: {:?}", affine_out);

            // --- x^T A + b ---
            let linear_out_vec = affine_out_vec
                .into_iter()
                .map(|affine_out| {
                    affine_out.into_iter().zip(
                        biases_mle.mle.iter()
                    ).map(|(x, bias)| {
                        x + bias
                    }).collect_vec()
                }).collect_vec();

            // println!("linear_out: {:?}", linear_out);

            let (mut relu_decomp_mle_vec, mut next_hidden_layer_vals_mle_vec) = (vec![], vec![]);
            
            for linear_out in linear_out_vec.into_iter() {
                // --- Compute 64-bit bin decomp of `linear_out`, as required for ReLU in circuit ---
                let relu_decomp: Vec<BinDecomp64Bit<F>> = linear_out.iter().map(|x| {
                    // println!("x: {:?}", x.get_lower_128());
                    let mut field_elem_i64_repr = x.get_lower_128() as i64;
                    if *x > F::from(1 << 63) {
                        // println!("x: {:?}", x.get_lower_128());
                        assert!(x.neg().get_lower_128() < (1<<63));
                        field_elem_i64_repr = -1 * (x.neg().get_lower_128() as i64); // maybe assert
                    }
                    // println!("field_elem_i64_repr: {:?}", field_elem_i64_repr);
                    let decomp = build_signed_bit_decomposition(field_elem_i64_repr, 64).unwrap();
                    BinDecomp64Bit::<F>::from(decomp)
                }).collect_vec();

                // println!("relu_decomp: {:?}", relu_decomp);

                let relu_decomp_mle = DenseMle::new_from_iter(
                    relu_decomp.clone().into_iter(),
                    LayerId::Input(0),
                    None,
                );

                // --- Compute ReLU function for next iteration of matmul inputs ---
                // let post_relu_bookkeeping_table = linear_out.into_iter().map(|x| {
                //     x.max(F::zero())
                // });
                // println!("relu_decomp_mle: {:?}", relu_decomp_mle);
                let post_relu_bookkeeping_table = compute_signed_pos_16_bit_recomp_from_64_bit_decomp(relu_decomp);
                // println!("post_relu_bookkeeping_table: {:?}", post_relu_bookkeeping_table);

                let next_hidden_layer_vals_mle = DenseMle::new_from_raw(post_relu_bookkeeping_table, LayerId::Input(0), None);

                relu_decomp_mle_vec.push(relu_decomp_mle);
                next_hidden_layer_vals_mle_vec.push(next_hidden_layer_vals_mle);
            }


            // println!("next_hidden_layer_vals_mle: {:?}", next_hidden_layer_vals_mle);
            let new_hidden_values_decomp_acc = hidden_values_decomp_acc
                .into_iter()
                .chain(std::iter::once(relu_decomp_mle_vec))
                .collect_vec();

            return (new_hidden_values_decomp_acc, next_hidden_layer_vals_mle_vec);
    });


    let final_val_combine = DenseMle::<F, F>::combine_mle_batch(final_val.clone());
    let last_weights = mlp_weights_biases.all_linear_weights_biases[mlp_weights_biases.all_linear_weights_biases.len() - 1].clone();

    // --- Construct matrices for x^T A ---
    let last_input_matrix = Matrix::new(
        final_val_combine.mle_ref(),
        batch_size as usize,
        last_weights.dim.in_features,
        final_val_combine.prefix_bits.clone(),
    );

    let weights_matrix = Matrix::new(
        last_weights.weights_mle.mle_ref(),
        last_weights.dim.in_features,
        last_weights.dim.out_features,
        last_weights.weights_mle.prefix_bits.clone(),
    );
    let affine_out = product_two_matrices(last_input_matrix, weights_matrix);

    let affine_out_vec = affine_out.chunks(batch_size)
    .map(|chunk| chunk.to_vec())
    .collect_vec();

    assert_eq!(affine_out_vec.len(), batch_size);
    assert_eq!(affine_out_vec[0].len(), last_weights.dim.out_features);

    // --- x^T A + b ---
    let final_out = affine_out_vec
        .into_iter()
        .map(|affine_out| {
            affine_out.into_iter().zip(
                last_weights.biases_mle.mle.iter()
            ).map(|(x, bias)| {
                x + bias
            }).collect_vec()
        }).collect_vec();

    assert_eq!(final_out.len(), batch_size);
    assert_eq!(final_out[0].len(), last_weights.dim.out_features);

    // println!("final_out: {:?}", final_out);

    DataparallelMLPInputData {
        input_mles,
        relu_bin_decomp_vecs: hidden_values_decomp,
        dim: DataparallelNNLinearInputDimension {
            batch_size,
            num_features,
        },
    }
}

/// Primary function which we care about!
pub fn load_dummy_mlp_input_and_weights<F: FieldExt>(
    input_dim: usize,
    hidden_dims: Option<Vec<usize>>,
    output_dim: usize,
) -> (
    MLPWeights<F>,
    MLPInputData<F>,
) {
    // --- Generate random inputs ---
    let input_mle = gen_random_mle(input_dim, 16);

    // --- Generate random weights/biases for all layers ---
    let all_input_dims = std::iter::once(input_dim)
        .chain(hidden_dims.clone().unwrap_or(vec![]));
    let all_output_dims = hidden_dims.unwrap_or(vec![])
        .into_iter()
        .chain(std::iter::once(output_dim));
    let all_linear_weights_biases = all_input_dims.zip(all_output_dims).map(|(input_dim, output_dim)| {
        NNLinearWeights {
            weights_mle: gen_random_mle(input_dim * output_dim, 16),
            biases_mle: gen_random_mle(output_dim, 16),
            dim: NNLinearDimension {
                in_features: input_dim,
                out_features: output_dim,
            },
        }
    }).collect_vec();

    let mlp_weights_biases = MLPWeights {
        all_linear_weights_biases,
    };

    // --- Generate bin decomps for all ReLU layers given inputs and weights ---
    let mlp_input_data = compute_hidden_layer_values_and_bin_decomps(
        &mlp_weights_biases,
        input_mle,
    );

    (mlp_weights_biases, mlp_input_data)
}

/// We assume that after every linear layer there is a non-linearity,
/// and therefore compute both the literal values and the bin decomp
/// for those values for every hidden layer.
pub fn compute_hidden_layer_values_and_bin_decomps<F: FieldExt>(
    mlp_weights_biases: &MLPWeights<F>,
    input_mle: DenseMle<F, F>,
) -> MLPInputData<F> {

    // --- Basically we are saving the intermediate outputs from BEFORE ReLU is computed ---
    let (hidden_values_decomp, final_val) = mlp_weights_biases.all_linear_weights_biases
        .clone()
        .into_iter()
        // --- No ReLU for last layer ---
        .rev()
        .skip(1)
        .rev()
        // ------------------------------
        .fold((vec![], input_mle.clone()), |(hidden_values_decomp_acc, last_layer_output), linear_weights_biases| {

            let NNLinearWeights { weights_mle, biases_mle, dim } = linear_weights_biases;

            // --- Construct matrices for x^T A ---
            let input_matrix = Matrix::new(
                last_layer_output.mle_ref(),
                1 as usize,
                last_layer_output.mle_ref().bookkeeping_table().len(),
                last_layer_output.prefix_bits.clone(),
            );
            let weights_matrix = Matrix::new(
                weights_mle.mle_ref(),
                dim.in_features,
                dim.out_features,
                weights_mle.prefix_bits.clone(),
            );
            let affine_out = product_two_matrices(input_matrix, weights_matrix);

            // println!("affine_out: {:?}", affine_out);

            // --- x^T A + b ---
            let linear_out = affine_out.into_iter().zip(
                biases_mle.mle.iter()
            ).map(|(x, bias)| {
                x + bias
            }).collect_vec();

            // println!("linear_out: {:?}", linear_out);

            // --- Compute 64-bit bin decomp of `linear_out`, as required for ReLU in circuit ---
            let relu_decomp: Vec<BinDecomp64Bit<F>> = linear_out.iter().map(|x| {
                // println!("x: {:?}", x.get_lower_128());
                let mut field_elem_i64_repr = x.get_lower_128() as i64;
                if *x > F::from(1 << 63) {
                    // println!("x: {:?}", x.get_lower_128());
                    assert!(x.neg().get_lower_128() < (1<<63));
                    field_elem_i64_repr = -1 * (x.neg().get_lower_128() as i64); // maybe assert
                }
                // println!("field_elem_i64_repr: {:?}", field_elem_i64_repr);
                let decomp = build_signed_bit_decomposition(field_elem_i64_repr, 64).unwrap();
                BinDecomp64Bit::<F>::from(decomp)
            }).collect_vec();

            // println!("relu_decomp: {:?}", relu_decomp);

            let relu_decomp_mle = DenseMle::new_from_iter(
                relu_decomp.clone().into_iter(),
                LayerId::Input(0),
                None,
            );

            // --- Compute ReLU function for next iteration of matmul inputs ---
            // let post_relu_bookkeeping_table = linear_out.into_iter().map(|x| {
            //     x.max(F::zero())
            // });
            // println!("relu_decomp_mle: {:?}", relu_decomp_mle);
            let post_relu_bookkeeping_table = compute_signed_pos_16_bit_recomp_from_64_bit_decomp(relu_decomp);
            // println!("post_relu_bookkeeping_table: {:?}", post_relu_bookkeeping_table);

            let next_hidden_layer_vals_mle = DenseMle::new_from_raw(post_relu_bookkeeping_table, LayerId::Input(0), None);

            // println!("next_hidden_layer_vals_mle: {:?}", next_hidden_layer_vals_mle);
            let new_hidden_values_decomp_acc = hidden_values_decomp_acc
                .into_iter()
                .chain(std::iter::once(relu_decomp_mle))
                .collect_vec();

            return (new_hidden_values_decomp_acc, next_hidden_layer_vals_mle);
    });


    // --- Construct matrices for x^T A ---
    let last_input_matrix = Matrix::new(
        final_val.mle_ref(),
        1 as usize,
        final_val.mle_ref().bookkeeping_table().len(),
        final_val.prefix_bits.clone(),
    );

    let last_weights = mlp_weights_biases.all_linear_weights_biases[mlp_weights_biases.all_linear_weights_biases.len() - 1].clone();
    let weights_matrix = Matrix::new(
        last_weights.weights_mle.mle_ref(),
        last_weights.dim.in_features,
        last_weights.dim.out_features,
        last_weights.weights_mle.prefix_bits.clone(),
    );
    let affine_out = product_two_matrices(last_input_matrix, weights_matrix);

    // --- x^T A + b ---
    let _final_out = affine_out.into_iter().zip(
        last_weights.biases_mle.mle.iter()
    ).map(|(x, bias)| {
        x + bias
    }).collect_vec();

    // println!("final_out: {:?}", final_out);

    MLPInputData {
        input_mle: input_mle.clone(),
        relu_bin_decomp: hidden_values_decomp,
        dim: NNLinearInputDimension {
            num_features: input_mle.mle_ref().bookkeeping_table().len(),
        },
    }
}

/// Converts signed 64-bit binary decomp into the recomposed absolute value of signed 16-bit value
pub fn compute_signed_pos_16_bit_recomp_from_64_bit_decomp<F: FieldExt>(bit_decomps_64_bit: Vec<BinDecomp64Bit<F>>) -> Vec<F> {

    const ORIG_BITWIDTH: usize = 64;
    const RECOMP_BITWIDTH: usize = 32;

    let result_iter = bit_decomps_64_bit.into_iter().map(
        |signed_bin_decomp| {

            // if the decomp is negative, return 0, as per relu
            if signed_bin_decomp.bits[ORIG_BITWIDTH-1] == F::one() {
                F::zero()
            } else {
                assert_eq!(signed_bin_decomp.bits[ORIG_BITWIDTH-1], F::zero());
                signed_bin_decomp.bits
                    .into_iter()
                    .take(RECOMP_BITWIDTH-1)
                    .enumerate()
                    .fold(F::zero(), |acc, (bit_idx, cur_bit)| {
                        let base = F::from(2_u64.pow(bit_idx as u32));
                        acc + base * cur_bit
                    })
            }
        }
    );

    result_iter.collect_vec()
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

/// Ryan's playground testing to ensure that the method works as intended
#[test]
pub fn test_partial_recomp() {
    let mut rng = test_rng();
    let test_arr = (0..10).map(|_| {
        let pos_val = rng.gen_range(0..2_i64.pow(31));
        if rng.gen_bool(0.5) {
            return -1 * pos_val;
        }
        return pos_val;
    }).collect_vec();

    // --- First generate 32-bit decomps of `test_arr` ---
    let bin_decomps_32_bit = test_arr.iter().map(|x| {
        build_signed_bit_decomposition(*x, 32).unwrap()
    }).collect_vec();

    // --- Next, perform 16-most-significant-bit recomps ---
    let recomp_bitwidth = 16;
    let start_bit_idx = 32 - recomp_bitwidth;
    let recomp_values = bin_decomps_32_bit.into_iter().map(|bin_decomp_32_bit| {
        // --- Iterate through the appropriate indices and recompose ---
        let recomp_value = bin_decomp_32_bit
            .clone()
            .into_iter()
            .skip(start_bit_idx)
            .rev()
            .skip(1)
            .enumerate()
            .fold(0_i64, |acc, (bit_idx, cur_bit)| {
                let base = 2_i64.pow((recomp_bitwidth - bit_idx - 2) as u32);
                let coeff = if cur_bit {1} else {0};
                acc + base * coeff
            });

        // --- If negative, return the negative version of `recomp_value` ---
        if *bin_decomp_32_bit.last().unwrap() {
            return -1 * recomp_value;
        }
        return recomp_value;
    }).collect_vec();

    // --- Compare against the "bit-shifted" version ---
    let bit_shifted_values = test_arr.iter().map(|x| {
        x >> start_bit_idx
    }).collect_vec();
    recomp_values.iter().zip(bit_shifted_values.iter()).for_each(|(x, y)| {
        if x != y {
            dbg!(format!("Yikes: {:?}, {:?}", x, y));
        }
    })
}

/// Loads the actual MNIST model weights from the given file path.
pub fn load_mnist_model_weights() -> () {
    todo!()
}

/// Loads the actual MNIST model inputs from the given file path.
pub fn load_mnist_input_data() -> () {
    todo!()
}