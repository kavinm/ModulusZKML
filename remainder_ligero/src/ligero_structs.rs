// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::utils::halo2_fft;
use crate::{def_labels, LcCommit, LcEncoding, LcEvalProof};
use fffft::FFTError;
use remainder_shared_types::FieldExt;

/// Ligero encoding
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LigeroEncoding<F: FieldExt> {
    /// number of inputs to the encoding
    pub orig_num_cols: usize,
    /// number of outputs from the encoding
    pub encoded_num_cols: usize,
    /// Code rate
    pub rho_inv: u8,
    /// For initialization purposes
    pub phantom: PhantomData<F>,
}

/// Total number of columns to be sent over
pub const N_COL_OPENS: usize = 128usize; // arbitrary, not secure
impl<F> LigeroEncoding<F>
where
    F: FieldExt,
{
    /// Grabs the matrix dimensions for M and M'
    pub fn get_dims(len: usize, rho: f64, ratio: f64) -> Option<(usize, usize, usize)> {
        // --- 0 < rho < 1 ---
        assert!(rho > 0f64);
        assert!(rho < 1f64);

        // compute #cols, which must be a power of 2 because of FFT
        // computes the encoded num cols that will get closest to the ratio for original num cols : num rows
        let encoded_num_cols =
            (((len as f64 * ratio).sqrt() / rho).ceil() as usize).checked_next_power_of_two()?;

        // minimize nr subject to #cols and rho
        // --- Not sure what the above is talking about, but basically computes ---
        // --- the other dimensions with respect to `encoded_num_cols` ---
        let orig_num_cols = (((encoded_num_cols as f64) * rho).floor()) as usize;
        let num_rows = (len + orig_num_cols - 1) / orig_num_cols;

        // --- Sanitycheck that we aren't going overboard or underboard ---
        assert!(orig_num_cols * num_rows >= len);
        assert!(orig_num_cols * (num_rows - 1) < len);

        Some((num_rows, orig_num_cols, encoded_num_cols))
    }

    fn _dims_ok(orig_num_cols: usize, encoded_num_cols: usize) -> bool {
        let sz = orig_num_cols < encoded_num_cols;
        let pow = encoded_num_cols.is_power_of_two();
        sz && pow
    }

    /// Creates a new Ligero encoding (data structure)
    pub fn new(len: usize, rho: f64, ratio: f64) -> Self {
        let rho_inv = (1.0 / rho) as u8;
        // --- Computes the matrix size for the commitment ---
        let (_, orig_num_cols, encoded_num_cols) = Self::get_dims(len, rho, ratio).unwrap();
        assert!(Self::_dims_ok(orig_num_cols, encoded_num_cols));
        Self {
            orig_num_cols,
            encoded_num_cols,
            rho_inv,
            phantom: PhantomData,
        }
    }

    /// Creates a new Ligero encoding from given dimensions
    pub fn new_from_dims(orig_num_cols: usize, encoded_num_cols: usize) -> Self {
        let rho_inv = (encoded_num_cols / orig_num_cols) as u8;
        assert!(Self::_dims_ok(orig_num_cols, encoded_num_cols));
        Self {
            orig_num_cols,
            encoded_num_cols,
            rho_inv,
            phantom: PhantomData,
        }
    }
}

impl<F> LcEncoding<F> for LigeroEncoding<F>
where
    F: FieldExt,
{
    type Err = FFTError;

    def_labels!(lcpc2d_test);

    fn encode(&self, inp: &mut [F]) -> Result<(), FFTError> {
        // --- So we need to convert num_cols(M) coefficients into num_cols(M) * (1 / rho) evaluations ---
        // --- All the coefficients past the original number of cols should be zero-padded ---
        debug_assert!(inp.iter().skip(self.orig_num_cols).all(|&v| v == F::zero()));

        // --- TODO!(ryancao): This is wasteful (we clone twice!!!) ---
        let evals = halo2_fft(
            inp.to_vec()
                .into_iter()
                .take(self.orig_num_cols)
                .collect_vec(),
            self.rho_inv,
        );
        inp.copy_from_slice(&evals[..]);

        Ok(())
    }

    fn get_dims(&self, len: usize) -> (usize, usize, usize) {
        let n_rows = (len + self.orig_num_cols - 1) / self.orig_num_cols;
        (n_rows, self.orig_num_cols, self.encoded_num_cols)
    }

    fn dims_ok(&self, orig_num_cols: usize, encoded_num_cols: usize) -> bool {
        let ok = Self::_dims_ok(orig_num_cols, encoded_num_cols);
        // let pc = encoded_num_cols == (1 << self.pc.get_log_len());
        let pc = true; // TODO!(ryancao): Fix this!!!
        let np = orig_num_cols == self.orig_num_cols;
        let nc = encoded_num_cols == self.encoded_num_cols;

        ok && pc && np && nc
    }

    fn get_n_col_opens(&self) -> usize {
        N_COL_OPENS
    }

    fn get_n_degree_tests(&self) -> usize {
        1
    }
}

/// Ligero commitment over LcCommit
pub type LigeroCommit<D, F> = LcCommit<D, LigeroEncoding<F>, F>;

/// Ligero evaluation proof over LcEvalProof
pub type LigeroEvalProof<D, E, F> = LcEvalProof<D, E, F>;
