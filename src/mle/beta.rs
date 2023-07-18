use std::{
    cmp::max,
    fmt::Debug,
    iter::{Cloned, Zip},
    marker::PhantomData,
};


use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_into_iter, rand::Rng, cfg_iter};
use itertools::{Itertools};
use rayon::{prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
}, slice::ParallelSlice};

use crate::FieldExt;
use crate::layer::Claim;

use super::{Mle, MleIndex, MleRef, MleAble};
use thiserror::Error;

#[derive(Error, Debug, Clone)]

/// Beta table struct for a product of mle refs
pub struct BetaTable<F: FieldExt> {
    layer_claim: Claim<F>,
    pub table: Vec<F>,
    relevant_indices: Vec<usize>,
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
}


impl<F: FieldExt> BetaTable<F> {

    /// Construct a new beta table using claims and a product of mle refs
    pub fn new(layer_claim: Claim<F>, mlerefs: &[impl MleRef<F = F>]) -> BetaTable<F> {
        let (layer_claims_idx, _) = &layer_claim;
        let (relevant_idx, relevant_claims): (Vec<usize>, Vec<(F, F)>) = mlerefs.iter()
        .map(|mleref| {
        mleref.get_mle_indices().iter().filter(
            // only want claims related to the indexed bits
            |mleindex| matches!(**mleindex, MleIndex::IndexedBit(_))
        ).map(|index| {
            // will panic if there are not enough claims from the previous layer because of direct indexing 
            if let MleIndex::IndexedBit(num) = index {
                (*num, (F::one() - layer_claims_idx[*num], layer_claims_idx[*num]))
            } else { // should never hit this
                (0, (F::one(), F::one()))
            }})
            }).flatten()
            .unzip();

        let unique_idx: Vec<usize> = relevant_idx.iter().unique().map(|idx| *idx).collect();
        let unique_claims: Vec<(F, F)> = relevant_claims.iter().unique().map(|claim| *claim).collect();

        let (one_minus_r, r) = unique_claims[0];
        let mut cur_table = vec![one_minus_r, r];
        
        // TODO!(vishruti) make this parallelizable
        for i in 1..unique_claims.len() {
            let (one_minus_r, r) = unique_claims[i];
            let mut firsthalf: Vec<F> = cfg_into_iter!(cur_table.clone()).map(|eval| eval * one_minus_r).collect();
            let secondhalf: Vec<F> = cfg_into_iter!(cur_table).map(|eval| eval * r).collect();
            firsthalf.extend(secondhalf.iter());
            cur_table = firsthalf;
        }

        BetaTable { layer_claim: layer_claim, table: cur_table, relevant_indices: unique_idx }   
    }

    /// Fix variable for a beta table
    pub fn beta_update(
        &mut self,
        round_index: usize,
        challenge: F,
    ) -> Result<(), BetaError> {
    
        let (layer_claims, _) = &self.layer_claim;
        let curr_beta = &self.table;
        let mut mult_factor = F::one();
        if self.relevant_indices.contains(&round_index) {
            let layer_claim = layer_claims[round_index];
            let layer_claim_inv = layer_claim.inverse().ok_or(BetaError::NoInverse)?;
            mult_factor = layer_claim_inv * (challenge * layer_claim + (F::one()-challenge)*(F::one()-layer_claim));
        }
        let new_beta: Vec<F> = cfg_into_iter!(curr_beta.clone()).skip(1).step_by(2).map(|curr_eval| curr_eval * mult_factor).collect();
        self.table = new_beta;
        
        Ok(())
    }
    
}