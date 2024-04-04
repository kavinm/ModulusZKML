// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use std::{fs, iter::repeat_with};

use ark_std::test_rng;
use itertools::{repeat_n, Itertools};
use rand::{prelude::Distribution, Rng};
use remainder_shared_types::{transcript::Transcript, FieldExt, Fr, Poseidon};

use crate::{
    layer::LayerId,
    mle::{dense::DenseMle, MleIndex},
    prover::Layers,
};

/// Returns a zero-padded version of `coeffs` with length padded
/// to the nearest power of two.
///
/// ## Arguments
///
/// * `coeffs` - The coefficients to be padded
///
/// ## Returns
///
/// * `padded_coeffs` - The coeffients, zero-padded to the nearest power of two (in length)
pub fn pad_to_nearest_power_of_two<F: FieldExt>(coeffs: Vec<F>) -> Vec<F> {
    // --- No need to duplicate things if we're already a power of two! ---
    if coeffs.len().is_power_of_two() {
        return coeffs;
    }

    let num_padding = coeffs.len().checked_next_power_of_two().unwrap() - coeffs.len();
    coeffs
        .into_iter()
        .chain(repeat_with(|| F::zero()).take(num_padding))
        .collect_vec()
}

/// Returns the argsort (i.e. indices) of the given vector slice.
///
/// Thanks ChatGPT!!!
pub fn argsort<T: Ord>(slice: &[T], invert: bool) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..slice.len()).collect();

    indices.sort_by(|&i, &j| {
        if invert {
            slice[j].cmp(&slice[i])
        } else {
            slice[i].cmp(&slice[j])
        }
    });

    indices
}

/// Helper function to create random MLE with specific number of vars
// pub fn get_random_mle<F: FieldExt>(num_vars: usize, rng: &mut impl Rng) -> DenseMle<F, F> {
pub fn get_random_mle<F: FieldExt>(num_vars: usize, rng: &mut impl Rng) -> DenseMle<F, F> {
    let capacity = 2_u32.pow(num_vars as u32);
    let bookkeeping_table = repeat_with(|| F::from(rng.gen::<u64>()))
        .take(capacity as usize)
        .collect_vec();
    DenseMle::new_from_raw(bookkeeping_table, LayerId::Input(0), None)
}

/// Helper function to create random MLE with specific number of vars
pub fn get_range_mle<F: FieldExt>(num_vars: usize) -> DenseMle<F, F> {
    // let mut rng = test_rng();
    let capacity = 2_u32.pow(num_vars as u32);
    let bookkeeping_table = (0..capacity)
        .map(|idx| F::from(idx as u64 + 1))
        .take(capacity as usize)
        .collect_vec();
    DenseMle::new_from_raw(bookkeeping_table, LayerId::Input(0), None)
}

/// Helper function to create random MLE with specific length
pub fn get_random_mle_with_capacity<F: FieldExt>(capacity: usize) -> DenseMle<F, F> {
    let mut rng = test_rng();
    let bookkeeping_table = repeat_with(|| F::from(rng.gen::<u64>()))
        .take(capacity)
        .collect_vec();
    DenseMle::new_from_raw(bookkeeping_table, LayerId::Input(0), None)
}

///returns an iterator that wil give permutations of binary bits of size num_bits
///
/// 0,0,0 -> 0,0,1 -> 0,1,0 -> 0,1,1 -> 1,0,0 -> 1,0,1 -> 1,1,0 -> 1,1,1
pub(crate) fn bits_iter<F: FieldExt>(num_bits: usize) -> impl Iterator<Item = Vec<MleIndex<F>>> {
    std::iter::successors(
        Some(vec![MleIndex::<F>::Fixed(false); num_bits]),
        move |prev| {
            let mut prev = prev.clone();
            let mut removed_bits = 0;
            for index in (0..num_bits).rev() {
                let curr = prev.remove(index);
                if curr == MleIndex::Fixed(false) {
                    prev.push(MleIndex::Fixed(true));
                    break;
                } else {
                    removed_bits += 1;
                }
            }
            if removed_bits == num_bits {
                None
            } else {
                Some(
                    prev.into_iter()
                        .chain(repeat_n(MleIndex::Fixed(false), removed_bits))
                        .collect_vec(),
                )
            }
        },
    )
}

/// Returns the specific bit decomp for a given index,
/// using `num_bits` bits. Note that this returns the
/// decomposition in BIG ENDIAN!
pub fn get_mle_idx_decomp_for_idx<F: FieldExt>(idx: usize, num_bits: usize) -> Vec<MleIndex<F>> {
    (0..(num_bits))
        .rev()
        .into_iter()
        .map(|cur_num_bits| {
            let is_one =
                (idx % 2_usize.pow(cur_num_bits as u32 + 1)) >= 2_usize.pow(cur_num_bits as u32);
            MleIndex::Fixed(is_one)
        })
        .collect_vec()
}

/// Returns whether a particular file exists in the filesystem
///
/// TODO!(ryancao): Shucks does this check a relative path...?
pub fn file_exists(file_path: &String) -> bool {
    match fs::metadata(file_path) {
        Ok(file_metadata) => file_metadata.is_file(),
        Err(_) => false,
    }
}

pub fn hash_layers<F: FieldExt, Tr: Transcript<F>>(layers: &Layers<F, Tr>) -> F {
    let mut sponge: Poseidon<F, 3, 2> = Poseidon::new(8, 57);

    layers.0.iter().for_each(|layer| {
        let item = format!("{}", layer.circuit_description_fmt());
        let bytes = item.as_bytes();
        let elements: Vec<F> = bytes
            .chunks(62)
            .map(|bytes| {
                let base = F::from(8);
                let first = bytes[0];
                bytes
                    .iter()
                    .skip(1)
                    .fold(
                        (F::from(first as u64), base.clone()),
                        |(accum, power), byte| {
                            let accum = accum + (F::from(byte.clone() as u64) * power);
                            let power = power * base;
                            (accum, power)
                        },
                    )
                    .0
            })
            .collect_vec();

        sponge.update(&elements);
    });

    sponge.squeeze()
}
