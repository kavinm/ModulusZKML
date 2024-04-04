// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//!Module for dealing with the Beta equality function

use std::fmt::Debug;

use ark_std::cfg_into_iter;
use itertools::Itertools;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::layer::{claims::Claim, LayerId};
use remainder_shared_types::FieldExt;

use super::{
    dense::{DenseMle, DenseMleRef},
    MleIndex, MleRef,
};
use thiserror::Error;

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
/// Beta table struct for a product of mle refs
pub struct BetaTable<F> {
    pub(crate) layer_claim_vars: Vec<F>,
    ///The bookkeeping table for the beta table
    /// TODO(Get rid of BetaTable's reliance on the DenseMleRef type; Create a shared subtype for the shared behavior)
    pub table: DenseMleRef<F>,
    pub relevant_indices: Vec<usize>,
}

/// Error handling for beta table construction
#[derive(Error, Debug, Clone)]
pub enum BetaError {
    #[error("claim index is 0, cannot take inverse")]
    ///claim index is 0, cannot take inverse
    NoInverse,
    #[error("not enough claims to compute beta table")]
    ///not enough claims to compute beta table
    NotEnoughClaims,
    #[error("cannot make beta table over empty mle list")]
    ///cannot make beta table over empty mle list
    EmptyMleList,
    #[error("cannot update beta table")]
    ///cannot update beta table
    BetaUpdateError,
    #[error("MLE bits were not indexed")]
    ///MLE bits were not indexed
    MleNotIndexedError,
    #[error("Beta table doesn't contain the particular indexed bit")]
    ///Beta table doesn't contain the particular indexed bit
    IndexedBitNotFoundError,
}

/// Computes \tilde{\beta}((x_1, ..., x_n), (y_1, ..., y_n))
///
/// Panics if `challenge_one` and `challenge_two` don't have
/// the same length!
pub fn compute_beta_over_two_challenges<F: FieldExt>(
    challenge_one: &Vec<F>,
    challenge_two: &Vec<F>,
) -> F {
    //assert_eq!(challenge_one.len(), challenge_two.len());

    // --- Formula is just \prod_i (x_i * y_i) + (1 - x_i) * (1 - y_i) ---
    let one = F::one();
    challenge_one
        .iter()
        .zip(challenge_two.iter())
        .fold(F::one(), |acc, (x_i, y_i)| {
            acc * ((*x_i * y_i) + (one - x_i) * (one - y_i))
        })
}

/// `fix_variable` for a beta table.
pub(crate) fn compute_new_beta_table<F: FieldExt>(
    beta_table: &BetaTable<F>,
    round_index: usize,
    challenge: F,
) -> Result<Vec<F>, BetaError> {
    let curr_beta = beta_table.table.bookkeeping_table();

    // --- This should always be true now, no? ---
    if beta_table.relevant_indices.contains(&round_index) {
        let layer_claim = beta_table.layer_claim_vars[round_index];
        let layer_claim_inv = layer_claim.invert();
        if layer_claim_inv.is_none().into() {
            return Err(BetaError::NoInverse);
        }
        // --- The below should be safe since we check for `is_none()` above ---
        let mult_factor = layer_claim_inv.unwrap()
            * (challenge * layer_claim + (F::one() - challenge) * (F::one() - layer_claim));

        let new_beta: Vec<F> = cfg_into_iter!(curr_beta)
            .skip(1)
            .step_by(2)
            .map(|curr_eval| *curr_eval * mult_factor)
            .collect();
        return Ok(new_beta);
    }

    Err(BetaError::IndexedBitNotFoundError)
}

/// Splits the beta table by the second most significant bit when we have nested selectors
/// (the case where the selector bit is not the independent variable)
pub(crate) fn beta_split<F: FieldExt>(
    beta_mle_ref: &DenseMleRef<F>,
) -> (DenseMleRef<F>, DenseMleRef<F>) {
    // the first split is to take two, then skip two (0, 1 mod 4)
    let beta_bookkeep_first: Vec<F> = beta_mle_ref
        .bookkeeping_table()
        .iter()
        .enumerate()
        .filter(|&(i, _)| (i % 4 == 0) | (i % 4 == 1))
        .map(|(_, v)| *v)
        .collect();

    // the other half -- (2, 3 mod 4)
    let beta_bookkeep_second: Vec<F> = beta_mle_ref
        .bookkeeping_table()
        .iter()
        .enumerate()
        .filter(|&(i, _)| (i % 4 == 2) | (i % 4 == 3))
        .map(|(_, v)| *v)
        .collect();

    let mut beta_first_mle_ref: DenseMleRef<F> =
        DenseMle::new_from_raw(beta_bookkeep_first, LayerId::Input(0), None).mle_ref();
    let mut beta_second_mle_ref: DenseMleRef<F> =
        DenseMle::new_from_raw(beta_bookkeep_second, LayerId::Input(0), None).mle_ref();

    beta_first_mle_ref.index_mle_indices(0);
    beta_second_mle_ref.index_mle_indices(0);

    (beta_first_mle_ref, beta_second_mle_ref)
}

impl<F: FieldExt> BetaTable<F> {
    /// Construct a new beta table using a single claim
    pub(crate) fn new(layer_claim_vars: Vec<F>) -> Result<BetaTable<F>, BetaError> {
        if layer_claim_vars.len() > 0 {
            let (one_minus_r, r) = (F::one() - layer_claim_vars[0], layer_claim_vars[0]);
            let mut cur_table = vec![one_minus_r, r];

            // TODO!(vishruti) make this parallelizable
            for claim in layer_claim_vars.iter().skip(1) {
                let (one_minus_r, r) = (F::one() - claim, claim);
                let mut firsthalf: Vec<F> = cfg_into_iter!(cur_table.clone())
                    .map(|eval| eval * one_minus_r)
                    .collect();
                let secondhalf: Vec<F> = cfg_into_iter!(cur_table).map(|eval| eval * r).collect();
                firsthalf.extend(secondhalf.iter());
                cur_table = firsthalf;
            }

            let iterated_bit_indices = (0..layer_claim_vars.len()).collect_vec();
            let cur_table_mle_ref: DenseMleRef<F> =
                DenseMle::new_from_raw(cur_table, LayerId::Input(0), None).mle_ref();
            Ok(BetaTable {
                layer_claim_vars,
                table: cur_table_mle_ref,
                relevant_indices: iterated_bit_indices,
            })
        } else {
            Ok(BetaTable {
                layer_claim_vars: vec![],
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
                relevant_indices: vec![],
            })
        }
    }

    /// Fix variable for a beta table
    pub(crate) fn beta_update(
        &mut self,
        round_index: usize,
        challenge: F,
    ) -> Result<(), BetaError> {
        // --- Use the pure function ---
        let new_beta = compute_new_beta_table(self, round_index, challenge);
        match new_beta {
            Err(e) => Err(e),
            Ok(new_beta_table) => {
                // --- If we successfully compute the new beta table, update the internal representation ---
                for mle_index in self.table.mle_indices.iter_mut() {
                    if *mle_index == MleIndex::IndexedBit(round_index) {
                        mle_index.bind_index(challenge);
                    }
                }
                self.table.bookkeeping_table = new_beta_table;
                self.table.num_vars -= 1;
                Ok(())
            }
        }
    }
}
