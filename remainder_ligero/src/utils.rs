// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use std::iter::repeat_with;

use crate::log2;
use ark_std::test_rng;
use halo2_proofs::poly::EvaluationDomain;
use rand::Rng;
use remainder_shared_types::FieldExt;

/// TODO!(ryancao): Add support for passing in an RNG rather than
/// constructing one within the function
pub fn get_random_coeffs_for_multilinear_poly<F: FieldExt>(ml_num_vars: usize) -> Vec<F> {
    let mut rng = test_rng();
    repeat_with(|| F::from(rng.gen::<u64>()))
        .take(2_usize.pow(ml_num_vars as u32))
        .collect()
}

/// Grabs the matrix dimensions for M and M'
///
/// ## Arguments
///
/// * `poly_len` - Number of coefficients in the actual polynomial
/// * `rho_inv` - rho^{-1}, i.e. the code rate
pub fn get_ligero_matrix_dims(
    poly_len: usize,
    rho_inv: u8,
    ratio: f64,
) -> Option<(usize, usize, usize)> {
    // --- Compute rho ---
    let rho: f64 = 1. / (rho_inv as f64);

    // --- 0 < rho < 1 ---
    assert!(rho > 0f64);
    assert!(rho < 1f64);

    // compute #cols, which must be a power of 2 because of FFT
    // computes the encoded num cols that will get closest to the ratio for original num cols : num rows
    let encoded_num_cols =
        (((poly_len as f64 * ratio).sqrt() / rho).ceil() as usize).checked_next_power_of_two()?;

    // minimize nr subject to #cols and rho
    // --- Not sure what the above is talking about, but basically computes ---
    // --- the other dimensions with respect to `encoded_num_cols` ---
    let orig_num_cols = ((encoded_num_cols as f64) * rho).floor() as usize;
    let num_rows = (poly_len + orig_num_cols - 1) / orig_num_cols;

    // --- Sanitycheck that we aren't going overboard or underboard ---
    assert!(orig_num_cols * num_rows >= poly_len);
    assert!(orig_num_cols * (num_rows - 1) < poly_len);

    Some((num_rows, orig_num_cols, encoded_num_cols))
}

/// Wrapper function over Halo2's FFT
///
/// ## Arguments
/// * `coeffs` -
/// * `rho_inv` -
///
/// ## Returns
/// * `evals` -
pub fn halo2_fft<F: FieldExt>(coeffs: Vec<F>, rho_inv: u8) -> Vec<F> {
    // --- Sanitycheck ---
    debug_assert!(coeffs.len().is_power_of_two());
    debug_assert!(rho_inv.is_power_of_two());

    let log_num_coeffs = log2(coeffs.len());
    let num_evals = coeffs.len() * (rho_inv as usize);
    debug_assert!(num_evals.is_power_of_two());

    // --- Note that `2^{j + 1}` is the total number of evaluations you actually want, and `2^k` is the number of coeffs ---
    // dbg!(rho_inv);
    // dbg!(coeffs.len());
    // dbg!(coeffs.len() as u32);
    let evaluation_domain: EvaluationDomain<F> =
        EvaluationDomain::new(rho_inv as u32, log_num_coeffs as u32);

    // --- Creates the polynomial in coeff form and performs the FFT ---
    let polynomial_coeff = evaluation_domain.coeff_from_vec(coeffs);
    let polynomial_eval_form = evaluation_domain.coeff_to_extended(&polynomial_coeff);
    debug_assert_eq!(polynomial_eval_form.len(), num_evals);

    polynomial_eval_form.to_vec()
}

/// Wrapper function over Halo2's FFT
///
/// ## Arguments
/// * `coeffs` -
/// * `rho_inv` -
///
/// ## Returns
/// * `evals` -
pub fn halo2_ifft<F: FieldExt>(evals: Vec<F>, rho_inv: u8) -> Vec<F> {
    // --- Sanitycheck ---
    debug_assert!(evals.len().is_power_of_two());
    debug_assert!(rho_inv.is_power_of_two());

    // let log_num_evals = log2(evals.len());
    let num_coeffs = (evals.len() as f64 / rho_inv as f64) as usize;
    debug_assert_eq!(num_coeffs * rho_inv as usize, evals.len());
    debug_assert!(num_coeffs.is_power_of_two());
    let log_num_coeffs = log2(num_coeffs);

    // --- Note that `2^{j + 1}` is the total number of evaluations you actually want, and `2^k` is the number of coeffs ---
    let evaluation_domain: EvaluationDomain<F> =
        EvaluationDomain::new(rho_inv as u32, log_num_coeffs as u32);

    // --- Creates the polynomial in coeff form and performs the FFT ---
    let mut polynomial_eval_form = evaluation_domain.empty_extended();
    polynomial_eval_form.copy_from_slice(&evals);
    let polynomial_coeff_form = evaluation_domain.extended_to_coeff(polynomial_eval_form);

    // --- The polynomial should only have degree evals / rho_inv ---
    polynomial_coeff_form
        .iter()
        .skip(num_coeffs)
        .for_each(|coeff| {
            debug_assert_eq!(*coeff, F::zero());
        });

    polynomial_coeff_form.to_vec()
}

/// Grabs the least significant bits from a byte vector stored in little-endian form!
///
/// ## Arguments
/// * `bytes` - Vector of individual bytes, with EACH byte stored in big-endian but ORDERED
///     in the vector in little-endian
/// * `num_bits` - Number of bits to grab from the "pure" little-endian representation
///     of the number
///
/// ## Returns
/// * `val` - The binary recomposition (interpreted in little-endian) of the `num_bits`
///     least significant bits of `bytes`
pub fn get_least_significant_bits_to_usize_little_endian(bytes: Vec<u8>, num_bits: usize) -> usize {
    bytes
        .into_iter()
        .enumerate()
        .fold(0_usize, |acc, (idx, byte)| {
            // --- Grab only some during the cutoff... ---
            if idx * 8 < num_bits && (idx + 1) * 8 > num_bits {
                let num_remaining = num_bits - idx * 8;
                (0..num_remaining).fold(acc, |inner_acc, inner_idx| {
                    let multiplier = 2_usize.pow((8 * idx + inner_idx) as u32);
                    let contributor = (((byte >> inner_idx) & 1) as usize) * multiplier;
                    inner_acc + contributor
                })

            // --- All bytes before the cutoff... ---
            } else if (idx + 1) * 8 <= num_bits {
                (0..8).fold(acc, |inner_acc, inner_idx| {
                    let multiplier = 2_usize.pow((8 * idx + inner_idx) as u32);
                    let contributor = (((byte >> inner_idx) & 1) as usize) * multiplier;
                    inner_acc + contributor
                })

            // --- And none after the cutoff ---
            } else {
                acc
            }
        })
}

#[cfg(test)]
mod test {

    use crate::utils::get_least_significant_bits_to_usize_little_endian;
    use ark_std::test_rng;
    use rand::Rng;
    use remainder_shared_types::{FieldExt, Fr};

    use std::ops::Range;

    #[test]
    fn test_get_least_significant_bits() {
        let mut rng = test_rng();

        // --- Everything should be equivalent to taking the remainder against 2^k ---
        (0..100).for_each(|_| {
            const VALUE_RANGE: std::ops::Range<u64> = 2_u64.pow(30)..2_u64.pow(40);
            let value = rng.gen_range::<u64, Range<u64>>(VALUE_RANGE);
            let fr_value = Fr::from(value);
            let num_cols = rng.gen_range::<usize, Range<usize>>(0..30);
            let expected_result = ((value as f64) % (2_usize.pow(num_cols as u32) as f64)) as usize;
            let value_le_bytes = fr_value.to_bytes().to_vec();
            let actual_result =
                get_least_significant_bits_to_usize_little_endian(value_le_bytes, num_cols);
            assert_eq!(expected_result, actual_result);
        })
    }
}
