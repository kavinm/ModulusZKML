// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//!Utilities involving the claims a layer makes

use ark_crypto_primitives::crh::sha256::digest::typenum::Or;
use itertools::Either;
use remainder_shared_types::transcript::{self};
use remainder_shared_types::FieldExt;
use tracing::instrument;

use crate::layer::combine_mle_refs::get_og_mle_refs;
use crate::mle::dense::{DenseMle, DenseMleRef};
use crate::mle::mle_enum::MleEnum;
use crate::mle::zero::ZeroMleRef;
use crate::mle::{MleIndex, MleRef};
use crate::prover::input_layer::enum_input_layer::InputLayerEnum;
use crate::prover::input_layer::InputLayer;
use crate::prover::{GKRError, ENABLE_OPTIMIZATION};
use crate::sumcheck::*;

use ark_std::{cfg_into_iter, cfg_iter};

use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use thiserror::Error;

use super::combine_mle_refs::CombineMleRefError;
use super::Layer;
use crate::layer::LayerId;

use serde::{Deserialize, Serialize};

use core::cmp::Ordering;
use std::cmp::max;
use std::fmt;

use log::{debug, info, warn};

use itertools::Itertools;

use ark_std::{end_timer, start_timer};

pub const ENABLE_REDUCED_WLX_EVALS: bool = true;
pub const ENABLE_CLAIM_GROUPING: bool = true;
pub const ENABLE_CLAIM_DEDUPLICATION: bool = true;
/// Controls raw mle eval vs expression eval
pub const ENABLE_RAW_MLE: bool = true;
/// Controls pre-fixing variables for shared columns, should only be true if above is true
pub const ENABLE_PRE_FIX: bool = true;

#[derive(Error, Debug, Clone)]
///Errors to do with aggregating and collecting claims
pub enum ClaimError {
    #[error("The Layer has not finished the sumcheck protocol")]
    ///The Layer has not finished the sumcheck protocol
    SumCheckNotComplete,
    #[error("MLE indices must all be fixed")]
    ///MLE indices must all be fixed
    ClaimMleIndexError,
    #[error("Layer ID not assigned")]
    ///Layer ID not assigned
    LayerMleError,
    #[error("MLE within MleRef has multiple values within it")]
    ///MLE within MleRef has multiple values within it
    MleRefMleError,
    #[error("Error aggregating claims")]
    ///Error aggregating claims
    ClaimAggroError,
    #[error("Should be evaluating to a sum")]
    ///Should be evaluating to a sum
    ExpressionEvalError,
    #[error("All claims in a group should agree on the number of variables")]
    NumVarsMismatch,
    #[error("All claims in a group should agree the destination layer field")]
    LayerIdMismatch,
    #[error("Error while combining mle refs in order to evaluate challenge point")]
    MleRefCombineError(CombineMleRefError),
}

/// A claim contains a `point` \in F^n along with the `result` \in F that an
/// associated layer MLE is expected to evaluate to. In other words, if `W : F^n
/// -> F` is the MLE, then the claim asserts: `W(point) == result`. It can
/// optionally maintain additional source/destination layer information through
/// `from_layer_id` and `to_layer_id`. This information can be used to speed up
/// claim aggregation.
#[derive(Clone, Serialize, Deserialize)]
pub struct Claim<F> {
    /// The point in F^n where the layer MLE is to be evaluated on.
    point: Vec<F>,
    /// The expected result of evaluating this layer's MLE on `point`.
    result: F,
    /// The layer ID of the layer that produced this claim (if present); origin
    /// layer.
    pub from_layer_id: Option<LayerId>,
    /// The layer ID of the layer containing the MLE this claim refers to (if
    /// present); destination layer.
    pub to_layer_id: Option<LayerId>,
    /// the mle ref associated with the claim
    pub mle_ref: Option<MleEnum<F>>,
}

impl<F: Clone> Claim<F> {
    /// Generate new raw claim without any origin/destination information.
    pub fn new_raw(point: Vec<F>, result: F) -> Self {
        Self {
            point,
            result,
            from_layer_id: None,
            to_layer_id: None,
            mle_ref: None,
        }
    }

    /// Generate new claim, potentially with origin/destination information.
    pub fn new(
        point: Vec<F>,
        result: F,
        from_layer_id: Option<LayerId>,
        to_layer_id: Option<LayerId>,
        mle_ref: Option<MleEnum<F>>,
    ) -> Self {
        Self {
            point,
            result,
            from_layer_id,
            to_layer_id,
            mle_ref,
        }
    }

    /// Returns the length of the `point` vector.
    pub fn get_num_vars(&self) -> usize {
        self.point.len()
    }

    /// Returns the point vector in F^n.
    pub fn get_point(&self) -> &Vec<F> {
        &self.point
    }

    /// Returns the expected result.
    pub fn get_result(&self) -> F {
        self.result.clone()
    }

    /// Returns the source Layer ID.
    pub fn get_from_layer_id(&self) -> Option<LayerId> {
        self.from_layer_id
    }

    /// Returns the destination Layer ID.
    pub fn get_to_layer_id(&self) -> Option<LayerId> {
        self.to_layer_id
    }
}

impl<F: fmt::Debug> fmt::Debug for Claim<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Claim")
            .field("point", &self.point)
            .field("result", &self.result)
            .field("from_layer_id", &self.from_layer_id)
            .field("to_layer_id", &self.to_layer_id)
            .finish()
    }
}

/// A collection of claims for the same layer with an API for accessing the
/// matrix of claim points in a multitude of ways (row-wise or column-wise).
/// This struct is useful for claim aggregation.
/// Invariant: All claims have to agree on `to_layer_id` and on the number of
/// variables.
#[derive(Clone, Debug)]
pub struct ClaimGroup<F> {
    /// A vector of claims in F^n.
    pub claims: Vec<Claim<F>>,
    /// TODO(Makis): The following fields are all redundant. We should remove
    /// them on the next refactoring and consider using iterators instead!
    /// -------------- REFACTOR NEEDED ---------------------
    /// The common layer ID of all claims stored in this group.
    src_layer_id: Option<LayerId>,
    /// A 2D matrix with the claim's points as its rows.
    claim_points_matrix: Vec<Vec<F>>,
    /// The points in `claims` is effectively a matrix of elements in F. We also
    /// store the transpose of this matrix for convenient access.
    claim_points_transpose: Vec<Vec<F>>,
    /// A vector of `self.get_num_claims()` elements. For each claim i,
    /// `result_vector[i]` stores the expected result of the i-th claim.
    result_vector: Vec<F>,
}

impl<F: Copy + Clone + std::fmt::Debug> ClaimGroup<F> {
    /// Builds a ClaimGroup<F> struct from a vector of claims. Also populates
    /// all the redundant fields for easy access to rows/columns.
    /// If the claims do not all agree on the number of variables, a
    /// `ClaimError::NumVarsMismatch` is returned.
    /// If the claims do not all agree on the `to_layer_id`, a
    /// `ClaimError::LayerIdMismatch` is returned.
    pub fn new(claims: Vec<Claim<F>>) -> Result<Self, ClaimError> {
        let num_claims = claims.len();
        if num_claims == 0 {
            return Ok(Self {
                claims: vec![],
                src_layer_id: None,
                claim_points_matrix: vec![],
                claim_points_transpose: vec![],
                result_vector: vec![],
            });
        }
        let num_vars = claims[0].get_num_vars();

        // Check all claims match on the number of variables.
        if !claims
            .clone()
            .into_iter()
            .all(|claim| claim.get_num_vars() == num_vars)
        {
            return Err(ClaimError::NumVarsMismatch);
        }

        // Check all claims match on the `to_layer_id` field.
        let layer_id = claims[0].get_to_layer_id();
        if !claims
            .clone()
            .into_iter()
            .all(|claim| claim.get_to_layer_id() == layer_id)
        {
            return Err(ClaimError::LayerIdMismatch);
        }

        // Populate the points_matrix
        let points_matrix: Vec<_> = claims
            .clone()
            .into_iter()
            .map(|claim| -> Vec<F> { claim.get_point().clone() })
            .collect();

        // Compute the claim points transpose.
        let claim_points_transpose: Vec<Vec<F>> = (0..num_vars)
            .map(|j| (0..num_claims).map(|i| claims[i].get_point()[j]).collect())
            .collect();

        // Compute the result vector.
        let result_vector: Vec<F> = (0..num_claims).map(|i| claims[i].get_result()).collect();

        Ok(Self {
            claims,
            src_layer_id: layer_id,
            claim_points_matrix: points_matrix,
            claim_points_transpose,
            result_vector,
        })
    }

    /// Returns the number of claims stored in this group.
    pub fn get_num_claims(&self) -> usize {
        self.claims.len()
    }

    /// Returns true if the group contains no claims.
    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    /// Returns the number of indices of the claims stored.
    /// Panics if no claims present.
    pub fn get_num_vars(&self) -> usize {
        self.claims[0].get_num_vars()
    }

    /// Returns the common destination layer ID of all the claims stored in this
    /// group.
    pub fn get_layer_id(&self) -> Option<LayerId> {
        self.src_layer_id
    }

    /// Returns the i-th result.
    /// # Panics
    /// If i is not in the range 0 <= i < `self.get_num_claims()`.
    pub fn get_result(&self, i: usize) -> F {
        self.claims[i].get_result()
    }

    /// Returns reference to the i-th point vector.
    /// # Panics
    /// When i is not in the range 0 <= i < `self.get_num_claims()`.
    pub fn get_challenge(&self, i: usize) -> &Vec<F> {
        &self.claims[i].get_point()
    }

    /// Returns a reference to a vector of `self.get_num_claims()` elements, the
    /// j-th entry of which is the i-th coordinate of the j-th claim's point. In
    /// other words, it returns the i-th column of the matrix containing the
    /// claim points as its rows.
    /// # Panics
    /// When i is not in the range: 0 <= i < `self.get_num_vars()`.
    pub fn get_points_column(&self, i: usize) -> &Vec<F> {
        &self.claim_points_transpose[i]
    }

    /// Returns a reference to an "m x n" matrix where n = `self.get_num_vars()`
    /// and m = `self.get_num_claims()` with the claim points as its rows.
    pub fn get_claim_points_matrix(&self) -> &Vec<Vec<F>> {
        &self.claim_points_matrix
    }

    /// Returns a reference to an "n x m" matrix where n = `self.get_num_vars()`
    /// and m = `self.get_num_claims()`, containing the claim points as its
    /// columns.
    pub fn get_claim_points_transpose(&self) -> &Vec<Vec<F>> {
        &self.claim_points_transpose
    }

    /// Returns a reference to a vector with m = `self.get_num_claims()`
    /// elements containing the results of all claims.
    pub fn get_results(&self) -> &Vec<F> {
        &self.result_vector
    }

    /// Returns a reference to the i-th claim.
    pub fn get_claim(&self, i: usize) -> &Claim<F> {
        &self.claims[i]
    }

    /// Returns a reference to a vector of claims contained in this group.
    pub fn get_claim_vector(&self) -> &Vec<Claim<F>> {
        &self.claims
    }

    /// Temporary hacky fix -- claims store their respective (unmodified)
    /// MleRefs instead of the layer itself. This simply looks at the list
    /// of all claims within a single ClaimGroup and grabs the associated
    /// MleRefs from there.
    pub fn get_claim_mle_refs(&self) -> Vec<MleEnum<F>> {
        self.claims
            .clone()
            .into_iter()
            .map(|claim| claim.mle_ref.unwrap())
            .collect_vec()
    }
}

// ---- Interface: Code outside `claims.rs` should be calling ----
// ---- `aggregate_claims` to perform claim aggregation. ----

/// Performs claim aggregation. Can be used by both the prover and the verifier.
/// * `claims`: a group of claims, all residing in the same layer (same
///   `to_layer_id` fields), to be aggregated into one.
/// * `compute_wlx_fn`: closure for computing the wlx evaluations. If
///   `aggregate_claims` is called by the prover, the closure should compute the
///   wlx evaluations, potentially using "smart" aggregation controlled by
///   `ENABLE_REDUCED_WLX_EVALS` which provides tighter bounds on the degree of
///   `W(l(x))`. A prover's `compute_wlx_fn` should never produce an error. If
///   called by the verifier, the closure should return the next wlx evaluations
///   received from the prover. In case claim aggregation requires more
///   evaluations than the ones provided by the prover, the closure should
///   return a `GKRError` which is propagated back to the caller of
///   `aggregate_claims`.
/// * `transcript`: is used to post wlx evaluations and generate challenges.
/// If successful, returns a pair containing the aggregated claim without
/// from/to layer ID information and a vector of wlx evaluations. The vector
/// either contains no evaluations (in the trivial case of aggregating a single
/// claim) or contains `k` vectors, the wlx evaluations produced during each of
/// the `k` naive claim aggregations performed.
/// TODO(Makis): Refactor this file to better expose the interface vs
/// implementation.
pub fn aggregate_claims<F: FieldExt>(
    claims: &ClaimGroup<F>,

    compute_wlx_fn: &mut impl FnMut(
        &ClaimGroup<F>,
        usize,
        Option<&Vec<MleEnum<F>>>,
    ) -> Result<Vec<F>, GKRError>,
    transcript: &mut impl transcript::Transcript<F>,
) -> Result<(Claim<F>, Vec<Vec<F>>), GKRError> {
    let num_claims = claims.get_num_claims();
    debug_assert!(num_claims > 0);
    info!("High-level claim aggregation on {num_claims} claims.");

    let claim_preproc_timer = start_timer!(|| format!("Claim preprocessing"));

    let layer_mle_refs = get_og_mle_refs(claims.get_claim_mle_refs());

    // Holds a sequence of relevant wlx evaluations, one for each claim
    // group that is being aggregated.
    let mut group_wlx_evaluations: Vec<Vec<F>> = vec![];

    let claims = preprocess_claims(claims.get_claim_vector().clone());
    let claim_groups = form_claim_groups(&claims);

    let num_claim_groups = claim_groups.len();

    debug!("Grouped claims for aggregation: ");
    for group in &claim_groups {
        debug!("GROUP:");
        for claim in group.get_claim_vector() {
            debug!("{:#?}", claim);
        }
    }

    end_timer!(claim_preproc_timer);
    let intermediate_timer = start_timer!(|| format!("Intermediate claim aggregation."));

    // TODO(Makis): Parallelize
    let intermediate_results: Result<Vec<(Claim<F>, Vec<Vec<F>>)>, GKRError> = claim_groups
        .into_iter()
        .enumerate()
        .map(|(idx, claim_group)| {
            aggregate_claims_in_one_round(
                &claim_group,
                &layer_mle_refs,
                compute_wlx_fn,
                idx,
                transcript,
            )
        })
        .collect();
    let intermediate_results = intermediate_results?;

    // TODO(Makis): Parallelize both
    let intermediate_claims = intermediate_results
        .clone()
        .into_iter()
        .map(|result| result.0)
        .collect();
    let mut intermediate_wlx_evals: Vec<Vec<F>> = intermediate_results
        .into_iter()
        .map(|result| result.1)
        .flatten()
        .collect();

    // Gather all wlx evaluations into one place.
    group_wlx_evaluations.append(&mut intermediate_wlx_evals);

    end_timer!(intermediate_timer);
    let final_timer = start_timer!(|| format!("Final stage aggregation."));

    // Finally, aggregate all intermediate claims.
    let (claim, mut wlx_evals_option) = aggregate_claims_in_one_round(
        &ClaimGroup::new(intermediate_claims).unwrap(),
        &layer_mle_refs,
        compute_wlx_fn,
        num_claim_groups, // Should be the final prover-supplied V_i(l(x)) evaluations
        transcript,
    )?;

    group_wlx_evaluations.append(&mut wlx_evals_option);

    end_timer!(final_timer);
    Ok((claim, group_wlx_evaluations))
}

// ---- Implementation: The following functions are used by ----
// ---- `aggregate_claims` and/or testing functions. ----

/// Aggregates a sequence of claim into a single point. If `claims` contains `m`
/// points `[u_0, u_1, ..., u_{m-1}]` where each `u_i \in F^n`, this function
/// computes a univariate polynomial vector `l : F -> F^n` such that `l(0) =
/// u_0, l(1) = u_1, ..., l(m-1) = u_{m-1}` using Lagrange interpolation, then
/// evaluates `l` on `r_star` and returns it.
/// # Requires
/// `claims_points` to be non-empty, otherwise a
/// `ClaimError::ClaimAggroError` is returned.
/// # TODO(Makis)
/// Using the ClaimGroup abstraction here is not ideal since we are only
/// operating on the points and not on the results. However, the ClaimGroup API
/// is convenient for accessing columns and makes the implementation more
/// readable. We should consider alternative designs.
#[instrument(level = "trace", err)]
#[instrument(level = "debug", skip_all, err)]
pub fn compute_aggregated_challenges<F: FieldExt>(
    claims: &ClaimGroup<F>,
    r_star: F,
) -> Result<Vec<F>, ClaimError> {
    if claims.is_empty() {
        return Err(ClaimError::ClaimAggroError);
    }

    let num_vars = claims.get_num_vars();

    // Compute r = l(r*) by performing Lagrange interpolation on each coordinate
    // using `evaluate_at_a_point`.
    let r: Vec<F> = cfg_into_iter!(0..num_vars)
        .map(|idx| {
            let evals = claims.get_points_column(idx);
            // Interpolate the value l(r*) from the values
            // l(0), l(1), ..., l(m-1) where m = # of claims.
            evaluate_at_a_point(evals, r_star).unwrap()
        })
        .collect();

    Ok(r)
}

/// Part of the claim aggregation process. It returns a vector of evaluations
/// [W(l(0)), W(l(1)), ..., W(l(k))] where W : F^n -> F is the layer MLE stored
/// in `layer`, l : F -> F^n is the interpolated polynomial on the claim points
/// in `claims` (see `compute_aggregated_challenges()` for a definition) and `k`
/// is *at least* the degree of the univariate polynomial W(l(x)) : F -> F.
/// A ClaimError::ClaimAggroError is returned if `claims` is empty.
/// TODO(Makis): Rename this function avoiding the term "wlx".
/// TODO(Makis): Due to the `ClaimGroup<F>` refactor, this function is now just
/// wrapper around `Layer<F>::get_wlx_evaluations()`. Consider removing it.
/*
pub(crate) fn compute_claim_wlx<F: FieldExt>(
    claims: &ClaimGroup<F>,
    layer: &impl Layer<F>,
) -> Result<Vec<F>, ClaimError> {
    if claims.is_empty() {
        return Err(ClaimError::ClaimAggroError);
    }
    let num_claims = claims.get_num_claims();
    let num_vars = claims.get_num_vars();
    let claim_mle_refs = claims.get_claim_mle_refs();

    let results = claims.get_results();
    let points_matrix = claims.get_claim_points_matrix();

    debug_assert_eq!(points_matrix.len(), num_claims);
    debug_assert_eq!(points_matrix[0].len(), num_vars);

    // Get the evals [W(l(0)), W(l(1)), ..., W(l(degree_upper_bound)) ]
    let wlx = layer.get_wlx_evaluations(points_matrix, results, claim_mle_refs, num_claims, num_vars)?;

    Ok(wlx)
}
*/

/// Sorts claims by `from_layer_id` to prepare them for grouping. Also performs
/// claim de-duplication if the `ENABLE_CLAIM_DEDUPLICATION` flag it set.
pub fn preprocess_claims<F: FieldExt>(mut claims: Vec<Claim<F>>) -> Vec<Claim<F>> {
    if !ENABLE_CLAIM_DEDUPLICATION && !ENABLE_CLAIM_GROUPING {
        // There is no need to sort the claims if no optimizations are enabled.
        return claims;
    }

    // Sort claims on the `from_layer_id` field.
    // A trivial total order is imposed which includes `None` values.
    claims.sort_by(|claim1, claim2| {
        match (claim1.get_from_layer_id(), claim2.get_from_layer_id()) {
            (Some(id1), Some(id2)) => match id1.cmp(&id2) {
                // Ties are broken by point value.
                // Ordering::Equal => claim1.get_point().cmp(claim2.get_point()),
                ordering => ordering,
            },
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            // (None, None) => claim1.get_point().cmp(claim2.get_point()),
            (None, None) => Ordering::Equal,
        }
    });

    // Perform claim de-duplication if flag is set.
    if !ENABLE_CLAIM_DEDUPLICATION {
        claims
    } else {
        info!("Performing claim de-duplication.");
        debug!("Num claims BEFORE dedup: {}", claims.len());
        // Remove duplicates.
        // TODO(Makis): Parallelize.
        let claims: Vec<Claim<F>> = claims
            .into_iter()
            .unique_by(|c| c.get_point().clone())
            .collect();
        debug!("\nNum claims AFTER dedup: {}", claims.len());
        claims
    }
}

/// Assign `claims` into groups to be aggregated together.  The naive version of
/// this function assigns all claims to a single group.  On the other hand, if
/// the `ENABLE_CLAIM_GROUPING` flag is set, it groups the claims based on the
/// `from_layer_id` field of each claim.
/// # Requires
/// All claims with the same `from_layer_id` should appear consecutively in the
/// `claims` vector. For example, `claims` can be sorted by `from_layer_id`.
pub fn form_claim_groups<F: FieldExt>(claims: &[Claim<F>]) -> Vec<ClaimGroup<F>> {
    info!("Forming claim group...");

    if !ENABLE_CLAIM_GROUPING {
        // Naive claim grouping: everything goes into the same group.
        vec![ClaimGroup::new(claims.to_vec()).unwrap()]
    } else {
        let num_claims = claims.len();
        let mut claim_group_vec: Vec<ClaimGroup<F>> = vec![];

        // Identify runs of claims with the same `from_layer_id` field.
        let mut start_index = 0;
        for idx in 1..num_claims {
            if claims[idx].get_from_layer_id() != claims[idx - 1].get_from_layer_id() {
                let end_index = idx;
                claim_group_vec
                    .push(ClaimGroup::new(claims[start_index..end_index].to_vec()).unwrap());
                start_index = idx;
            }
        }

        // Process the last group.
        let end_index = num_claims;
        claim_group_vec.push(ClaimGroup::new(claims[start_index..end_index].to_vec()).unwrap());

        claim_group_vec
    }
}

/// Low-level analogue of `aggregate_claims` which performs claim aggregation on
/// the claim group `claims` in a single stage without further grouping.
/// * `claims`: the group of claims to be aggregated.
/// * `compute_wlx_fn`: closure for computing the wlx evaluations. If
///   `aggregate_claims_in_one_round` is called by the prover, the closure
///   should compute the wlx evaluations, potentially using "smart" aggregation
///   controlled by `ENABLE_REDUCED_WLX_EVALS` which provides tighter bounds on
///   the degree of `W(l(x))`. A prover's `compute_wlx_fn` should never produce
///   an error. If called by the verifier, the closure should return the wlx
///   evaluations sent by the prover. In case claim aggregation requires more
///   evaluations than the ones provided by the prover, the closure should
///   return a `GKRError` which is propagated back to the caller of
///   `aggregate_claims_in_one_round`.
/// * `transcript`: is used to post wlx evaluations and generate challenges.
/// If successful, returns a pair containing the aggregated claim without
/// from/to layer ID information and a vector of wlx evaluations. The vector
/// either contains no evaluations (in the trivial case of aggregating a single
/// claim) or contains a single vector of the wlx evaluations produced during
/// this 1-step claim aggregation.
pub fn aggregate_claims_in_one_round<F: FieldExt>(
    claims: &ClaimGroup<F>,
    layer_mle_refs: &Vec<MleEnum<F>>,

    compute_wlx_fn: &mut impl FnMut(
        &ClaimGroup<F>,
        usize,
        Option<&Vec<MleEnum<F>>>,
    ) -> Result<Vec<F>, GKRError>,
    prover_supplied_wlx_group_idx: usize,
    transcript: &mut impl transcript::Transcript<F>,
) -> Result<(Claim<F>, Vec<Vec<F>>), GKRError> {
    let num_claims = claims.get_num_claims();
    debug_assert!(num_claims > 0);
    info!("Low-level claim aggregation on {num_claims} claims.");

    // Do nothing if there is only one claim.
    if num_claims == 1 {
        debug!("Received 1 claim. Doing nothing.");
        // Return the claim but erase any from/to layer info so as not to
        // trigger any checks from claim groups used in claim aggregation.
        let claim = Claim {
            from_layer_id: None,
            to_layer_id: None,
            ..claims.get_claim(0).clone()
        };

        return Ok((claim, vec![vec![]]));
    }

    // Aggregate claims by performing the claim aggregation protocol.
    // First compute V_i(l(x)).

    let wlx_evaluations =
        compute_wlx_fn(claims, prover_supplied_wlx_group_idx, Some(layer_mle_refs))?;
    let relevant_wlx_evaluations = wlx_evaluations[num_claims..].to_vec();

    // Append evaluations to the transcript before sampling a challenge.
    transcript
        .append_field_elements(
            "Claim Aggregation Wlx_evaluations",
            &relevant_wlx_evaluations,
        )
        .unwrap();

    // Next, sample `r^\star` from the transcript.
    let agg_chal = transcript
        .get_challenge("Challenge for claim aggregation")
        .unwrap();
    debug!("Aggregate challenge: {:#?}", agg_chal);

    let aggregated_challenges = compute_aggregated_challenges(claims, agg_chal).unwrap();
    let claimed_val = evaluate_at_a_point(&wlx_evaluations, agg_chal).unwrap();

    debug!("Aggregating claims: ");
    for c in claims.get_claim_vector() {
        debug!("   {:#?}", c);
    }

    debug!(
        "Low level aggregated claim:\n{:#?}",
        Claim::new_raw(aggregated_challenges.clone(), claimed_val)
    );

    Ok((
        Claim::new_raw(aggregated_challenges, claimed_val),
        vec![relevant_wlx_evaluations],
    ))
}

/// Returns an upper bound on the number of evaluations needed to represent the
/// polynomial `P(x) = W(l(x))` where `W : F^n -> F` is a multilinear polynomial
/// on `n` variables and `l : F -> F^n` is such that:
///  * `l(0) = claim_vecs[0]`,
///  *  `l(1) = `claim_vecs[1]`,
///  *   ...,
///  *  `l(m-1) = `claim_vecs[m-1]`.
/// It is guaranteed that the returned value is at least `num_claims =
/// claim_vecs.len()`.
/// # Panics
///  if `claim_vecs` is empty.
pub fn get_num_wlx_evaluations<F: FieldExt>(
    claim_vecs: &Vec<Vec<F>>,
) -> (usize, Option<Vec<usize>>) {
    let num_claims = claim_vecs.len();
    let num_vars = claim_vecs[0].len();

    if !ENABLE_REDUCED_WLX_EVALS {
        // A naive upper bound on the number of evaluations derived as follows:
        // Let n = `num_vars` and m = `num_claims`. Each monomial in W(x1, ...,
        // x_n) contains up to n variables, each of which will be substituted by
        // `l_i(x)`, a canonical line passing through m points and hence of
        // degree up to m-1, resulting in a polynomial W(l(x)) of degree at most
        // n * (m-1) which needs at least n * (m-1) + 1 evaluations to be fully
        // specified.
        debug!("Dummy num_evals");
        (max(num_vars * (num_claims - 1) + 1, num_claims), None)
    } else {
        debug!("Smart num_evals");
        let mut num_constant_columns = num_vars as i64;
        let mut common_idx = vec![];
        for j in 0..num_vars {
            let mut degree_reduced = true;
            for i in 1..num_claims {
                if claim_vecs[i][j] != claim_vecs[i - 1][j] {
                    num_constant_columns -= 1;
                    degree_reduced = false;
                    break;
                }
            }
            if degree_reduced {
                common_idx.push(j);
            }
        }
        assert!(num_constant_columns >= 0);
        debug!("degree_reduction = {}", num_constant_columns);

        // Evaluate the P(x) := W(l(x)) polynomial at deg(P) + 1
        // points. W : F^n -> F is a multi-linear polynomial on
        // `num_vars` variables and l : F -> F^n is a canonical
        // polynomial passing through `num_claims` points so its degree is
        // at most `num_claims - 1`. This imposes an upper
        // bound of `num_vars * (num_claims - 1)` to the degree of P.
        // However, the actual degree of P might be lower.
        // For any coordinate `i` such that all claims agree
        // on that coordinate, we can quickly deduce that `l_i(x)` is a
        // constant polynomial of degree zero instead of `num_claims -
        // 1` which brings down the total degree by the same amount.
        let num_evals =
            (num_vars) * (num_claims - 1) + 1 - (num_constant_columns as usize) * (num_claims - 1);
        debug!("num_evals originally = {}", num_evals);
        (max(num_evals, num_claims), Some(common_idx))
    }
}

/// Aggregates `claims` into a single claim on challenge point
/// `r_star` given the `wlx` evaluations of W(l(x)).
/// This function is used by the verifier in the process of verifying
/// claim aggregation.
/*
#[instrument(level = "trace", err)]
#[instrument(level = "debug", skip_all, err)]
pub(crate) fn verify_aggregate_claim<F: FieldExt>(
    wlx: &Vec<F>, // synonym for q(x)
    claims: &ClaimGroup<F>,
    r_star: F,
) -> Result<Claim<F>, ClaimError> {
    if claims.is_empty() {
        return Err(ClaimError::ClaimAggroError);
    }



    let num_claims = claims.get_num_claims();

    // Check q(0), q(1) equals the claimed value -- i.e. W(l(0)),
    // W(l(1)), etc.
    for idx in 0..num_claims {
        if claims.get_result(idx) != wlx[idx] {
            return Err(ClaimError::ClaimAggroError);
        }
    }

    let r = compute_aggregated_challenges(claims, r_star)?;
    let q_rstar = evaluate_at_a_point(wlx, r_star).unwrap();

    let aggregated_claim: Claim<F> = Claim::new(r, q_rstar, claims.get_layer_id(), None, None);
    Ok(aggregated_claim)
}
*/

#[cfg(test)]
// Makis: Making this public so that I can access some of the helper functions
// from "sumcheck/tests.rs".
pub(crate) mod tests {
    use crate::expression::{Expression, ExpressionStandard};
    use crate::layer::{from_mle, GKRLayer, LayerId};
    use crate::mle::dense::DenseMle;
    use rand::Rng;
    use remainder_shared_types::transcript::{poseidon_transcript::PoseidonTranscript, Transcript};

    use super::*;
    use ark_std::test_rng;
    use remainder_shared_types::Fr;

    #[test]
    fn test_get_claim() {
        // [1, 1, 1, 1] \oplus (1 - (1 * (1 + V[1, 1, 1, 1]))) * 2
        let expression1: ExpressionStandard<Fr> = ExpressionStandard::Constant(Fr::one());
        let mle = DenseMle::<_, Fr>::new_from_raw(
            vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()],
            LayerId::Input(0),
            None,
        );
        let expression3 = ExpressionStandard::Mle(mle.mle_ref());
        let expression = expression1.clone() + expression3.clone();
        // let expression = expression1.clone() * expression;
        let expression = expression1 - expression;
        let expression = expression * Fr::from(2);
        let _expression = expression3.concat_expr(expression);

        // TODO(ryancao): Need to create a layer and fix all the MLE variables...
    }

    // ------- Helper functions for claim aggregation tests -------

    /// Builds `ClaimGroup<Fr>` by evaluation an expression `expr` on
    /// each point in `points`.
    fn claims_from_expr_and_points(
        expr: &ExpressionStandard<Fr>,
        points: &Vec<Vec<Fr>>,
    ) -> ClaimGroup<Fr> {
        let claims_vector: Vec<Claim<Fr>> = cfg_into_iter!(points)
            .map(|point| {
                let mut exp = expr.clone();
                exp.index_mle_indices(0);
                let result = exp.evaluate_expr(point.clone()).unwrap();
                Claim::new_raw(point.clone(), result)
            })
            .collect();
        ClaimGroup::new(claims_vector).unwrap()
    }

    /// Builds GKR layer whose MLE is the function whose evaluations
    /// on the boolean hypercube are given by `mle_evals`.
    fn layer_from_evals(mle_evals: Vec<Fr>) -> GKRLayer<Fr, PoseidonTranscript<Fr>> {
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_evals, LayerId::Input(0), None);

        let layer = from_mle(
            mle,
            |mle| mle.mle_ref().expression(),
            |_, _, _| unimplemented!(),
        );

        let layer: GKRLayer<_, PoseidonTranscript<_>> = GKRLayer::new(layer, LayerId::Input(0));

        layer
    }

    /// Returns a random MLE expression with an associated GKR layer.
    fn build_random_mle_layer(num_vars: usize) -> GKRLayer<Fr, PoseidonTranscript<Fr>> {
        let mut rng = test_rng();
        let mle_evals: Vec<Fr> = (0..num_vars).map(|_| Fr::from(rng.gen::<u64>())).collect();
        layer_from_evals(mle_evals)
    }

    fn compute_claim_wlx<F: FieldExt>(claims: &ClaimGroup<F>, layer: &impl Layer<F>) -> Vec<F> {
        let num_claims = claims.get_num_claims();
        let num_vars = claims.get_num_vars();

        let results = claims.get_results();
        let points_matrix = claims.get_claim_points_matrix();

        debug_assert_eq!(points_matrix.len(), num_claims);
        debug_assert_eq!(points_matrix[0].len(), num_vars);

        // Get the evals [W(l(0)), W(l(1)), ..., W(l(degree_upper_bound)) ]

        let claim_mle_refs = claims.get_claim_mle_refs();
        layer
            .get_wlx_evaluations(points_matrix, results, claim_mle_refs, num_claims, num_vars)
            .unwrap()
    }

    /// Wraps around low-level claim aggregation WITHOUT Layer ID
    /// information.
    fn claim_aggregation_back_end_wrapper(
        layer: &impl Layer<Fr>,
        claims: &ClaimGroup<Fr>,
        r_star: Fr,
    ) -> Claim<Fr> {
        let r = compute_aggregated_challenges(claims, r_star).unwrap();
        let wlx = compute_claim_wlx(claims, layer);
        let claimed_val = evaluate_at_a_point(&wlx, r_star).unwrap();
        Claim::new_raw(r, claimed_val)
    }

    /// Compute l* = l(r*).
    fn compute_l_star(claims: &ClaimGroup<Fr>, r_star: &Fr) -> Vec<Fr> {
        let num_vars = claims.get_num_vars();

        (0..num_vars)
            .map(|i| {
                let evals: &Vec<Fr> = claims.get_points_column(i);
                evaluate_at_a_point(evals, r_star.clone()).unwrap()
            })
            .collect()
    }

    /// Wraps around high-level claim aggregation with Layer ID
    /// information.
    pub(crate) fn claim_aggregation_testing_wrapper(
        layer: &impl Layer<Fr>,
        claims: &ClaimGroup<Fr>,
    ) -> (Claim<Fr>, Vec<Vec<Fr>>) {
        let mut transcript = PoseidonTranscript::<Fr>::new("Dummy transcript for testing");
        aggregate_claims(
            claims,
            &mut |claim, _, mle_refs| Ok(compute_claim_wlx(claims, layer)),
            &mut transcript,
        )
        .unwrap()
    }

    // Returns expected aggregated claim of `expr` on l(r_star) = `l_star`.
    fn compute_expected_claim(
        layer: &GKRLayer<Fr, PoseidonTranscript<Fr>>,
        l_star: &Vec<Fr>,
    ) -> Claim<Fr> {
        let mut expr = layer.expression().clone();
        expr.index_mle_indices(0);
        let result = expr.evaluate_expr(l_star.clone()).unwrap();
        Claim::new_raw(l_star.clone(), result)
    }

    // To run this, we need to be able to control the optimization flags which
    // are currently constants.
    /*
    #[test]
    fn test_aggro_claim_smart_merge1() {
        // MLE on 3 variables (2^3 = 8 evals)
        let mle_evals: Vec<Fr> = vec![1, 2, 42, 4, 5, 6, 7, 17]
            .into_iter()
            .map(Fr::from)
            .collect();
        let points = vec![
            vec![Fr::from(1), Fr::from(2), Fr::from(3)],
            vec![Fr::from(2), Fr::from(2), Fr::from(3)],
            vec![Fr::from(4), Fr::from(5), Fr::from(6)],
            vec![Fr::from(5), Fr::from(5), Fr::from(6)],
        ];
        let r_star = Fr::from(10);

        // ---------------

        let layer = layer_from_evals(mle_evals);
        let mut claims = claims_from_expr_and_points(layer.expression(), &points);

        // ---- FOR TESTING ONLY ----
        claims.claims[0].from_layer_id = Some(LayerId::Layer(0));
        claims.claims[1].from_layer_id = Some(LayerId::Layer(0));
        claims.claims[2].from_layer_id = Some(LayerId::Layer(1));
        claims.claims[3].from_layer_id = Some(LayerId::Layer(1));
        // --------------------------

        // W(l(0)), W(l(1)) computed by hand.
        assert_eq!(claims.get_result(0), Fr::from(72));
        assert_eq!(claims.get_result(1), Fr::from(283));
        assert_eq!(claims.get_result(2), Fr::from(4044));
        assert_eq!(claims.get_result(3), Fr::from(5290));

        let l_star = compute_l_star(&claims, &r_star);

        // Compare to l(10) computed by hand.
        assert_eq!(l_star, vec![Fr::from(11), Fr::from(13), Fr::from(5)]);

        let wlx = compute_claim_wlx(&claims, &layer);
        // assert_eq!(wlx, vec![Fr::from(163), Fr::from(1015), Fr::from(2269)]);

        let aggregated_claim = claim_aggregation_testing_wrapper(&layer, &claims);
        let expected_claim = compute_expected_claim(&layer, &l_star);

        // Compare to W(l_star) computed by hand.
        // assert_eq!(expected_claim.get_result(), Fr::from(26773));

        assert_eq!(aggregated_claim.0.get_result(), expected_claim.get_result());
    }
    */

    // TODO(Makis): Fix this outdated test.
    /*
    #[test]
    fn test_verify_claim_aggro() {
        let mle_v1 = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_v1, LayerId::Input(0), None);
        let mle_ref = mle1.mle_ref();
        let mut expr = ExpressionStandard::Mle(mle_ref);
        let _expr_copy = expr.clone();

        let layer = from_mle(
            mle1,
            |mle| mle.mle_ref().expression(),
            |_, _, _| unimplemented!(),
        );
        let layer: GKRLayer<_, PoseidonTranscript<_>> = GKRLayer::new(layer, LayerId::Input(0));

        let chals1 = vec![Fr::from(1), Fr::from(2)];
        let chals2 = vec![Fr::from(2), Fr::from(3)];
        let chals3 = vec![Fr::from(3), Fr::from(1)];
        let chals = vec![&chals1, &chals2, &chals3];
        let mut valchal: Vec<Fr> = Vec::new();

        for i in 0..3 {
            let mut exp = expr.clone();
            exp.index_mle_indices(0);
            let eval = exp.evaluate_expr((*chals[i]).clone());
            valchal.push(eval.unwrap());
        }

        let claim1: Claim<Fr> = Claim::new_raw(chals1, valchal[0]);
        let claim2: Claim<Fr> = Claim::new_raw(chals2, valchal[1]);
        let claim3: Claim<Fr> = Claim::new_raw(chals3, valchal[2]);
        let claims = vec![claim1, claim2, claim3];

        let rchal = Fr::from(2).neg();

        let claim_group = ClaimGroup::new(claims).unwrap();
        let (res, wlx) = claim_aggregation_testing_wrapper(&layer, &claim_group);
        let rounds = expr.index_mle_indices(0);
        // let verify_result = verify_aggregate_claim(&wlx.unwrap(), &claim_group, rchal).unwrap();

        // Makis; This test isn't verifying anything apart form the
        // fact that no errors are produced :/
    }
    */
}
