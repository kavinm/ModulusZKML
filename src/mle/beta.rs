use std::{
    cmp::max,
    fmt::Debug,
    iter::{Cloned, Zip},
    marker::PhantomData,
};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_into_iter, cfg_iter, rand::Rng};
use itertools::Itertools;
use rayon::{
    prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};

use crate::layer::Claim;
use crate::FieldExt;

use super::{
    dense::{DenseMle, DenseMleRef},
    Mle, MleAble, MleIndex, MleRef,
};
use thiserror::Error;

#[derive(Error, Debug, Clone)]

/// Beta table struct for a product of mle refs
pub struct BetaTable<F: FieldExt> {
    layer_claim: Claim<F>,
    pub table: DenseMleRef<F>,
    relevant_indices: Vec<usize>,
    pub indexed: bool,
}

/// Error handling for beta table construction
#[derive(Error, Debug, Clone)]
pub enum BetaError {
    #[error("claim index is 0, cannot take inverse")]
    NoInverse,
    #[error("not enough claims to compute beta table")]
    NotEnoughClaims,
    #[error("cannot make beta table over empty mle list")]
    EmptyMleList,
    #[error("cannot update beta table")]
    BetaUpdateError,
    #[error("MLE bits were not indexed")]
    MleNotIndexedError,
    #[error("Beta table doesn't contain the particular indexed bit")]
    IndexedBitNotFoundError,
}

/// fully evaluate a beta table
pub fn evaluate_beta<F: FieldExt>(
    beta_table: &mut BetaTable<F>,
    challenges: Vec<F>,
) -> Result<F, BetaError> {
    challenges
        .into_iter()
        .enumerate()
        .for_each(|(round_idx, challenge)| {
            let _ = beta_table.beta_update(round_idx, challenge);
        });
    let beta_bt = beta_table.table.bookkeeping_table();
    if beta_bt.len() == 1 {
        return Ok(beta_bt[0]);
    }
    Err(BetaError::BetaUpdateError)
}

/// `fix_variable` for a beta table.
pub fn compute_new_beta_table<F: FieldExt>(
    beta_table: &BetaTable<F>,
    round_index: usize,
    challenge: F,
) -> Result<Vec<F>, BetaError> {
    let (layer_claims, _) = &beta_table.layer_claim;
    let curr_beta = beta_table.table.bookkeeping_table().clone();

    // --- This should always be true now, no? ---
    if beta_table.relevant_indices.contains(&round_index) {
        let layer_claim = layer_claims[round_index];
        let layer_claim_inv = layer_claim.inverse().ok_or(BetaError::NoInverse)?;
        let mult_factor = layer_claim_inv
            * (challenge * layer_claim + (F::one() - challenge) * (F::one() - layer_claim));

        let new_beta: Vec<F> = cfg_into_iter!(curr_beta.clone())
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
pub fn beta_split<F: FieldExt>(beta_mle_ref: &DenseMleRef<F>) -> (DenseMleRef<F>, DenseMleRef<F>) {
    // the first split is to take two, then skip two (0, 1 mod 4)
    let beta_bookkeep_first: Vec<F> = beta_mle_ref
        .bookkeeping_table()
        .into_iter()
        .enumerate()
        .filter(|&(i, _)| (i % 4 == 0) | (i % 4 == 1))
        .map(|(_, v)| *v)
        .collect();

    // the other half -- (2, 3 mod 4)
    let beta_bookkeep_second: Vec<F> = beta_mle_ref
        .bookkeeping_table()
        .into_iter()
        .enumerate()
        .filter(|&(i, _)| (i % 4 == 2) | (i % 4 == 3))
        .map(|(_, v)| *v)
        .collect();

    let mut beta_first_mle_ref: DenseMleRef<F> = DenseMle::new(beta_bookkeep_first).mle_ref();
    let mut beta_second_mle_ref: DenseMleRef<F> = DenseMle::new(beta_bookkeep_second).mle_ref();

    beta_first_mle_ref.index_mle_indices(0);
    beta_second_mle_ref.index_mle_indices(0);

    (beta_first_mle_ref, beta_second_mle_ref)
}

impl<F: FieldExt> BetaTable<F> {
    /// Construct a new beta table using a single claim
    pub fn new(layer_claim: Claim<F>) -> Result<BetaTable<F>, BetaError> {
        let (layer_claim_vars, _) = &layer_claim;
        let (one_minus_r, r) = (F::one() - layer_claim_vars[0], layer_claim_vars[0]);
        let mut cur_table = vec![one_minus_r, r];

        // TODO!(vishruti) make this parallelizable
        for i in 1..layer_claim_vars.len() {
            let (one_minus_r, r) = (F::one() - layer_claim_vars[i], layer_claim_vars[i]);
            let mut firsthalf: Vec<F> = cfg_into_iter!(cur_table.clone())
                .map(|eval| eval * one_minus_r)
                .collect();
            let secondhalf: Vec<F> = cfg_into_iter!(cur_table).map(|eval| eval * r).collect();
            firsthalf.extend(secondhalf.iter());
            cur_table = firsthalf;
        }

        let iterated_bit_indices = (0..layer_claim_vars.len()).into_iter().collect_vec();
        let cur_table_mle_ref: DenseMleRef<F> = DenseMle::new(cur_table).mle_ref();
        Ok(BetaTable {
            layer_claim,
            table: cur_table_mle_ref,
            relevant_indices: iterated_bit_indices,
            indexed: false,
        })
    }

    /// Fix variable for a beta table
    pub fn beta_update(&mut self, round_index: usize, challenge: F) -> Result<(), BetaError> {
        // --- Use the pure function ---
        let new_beta = compute_new_beta_table(&self, round_index, challenge);
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
