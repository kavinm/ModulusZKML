//!A Layer with 0 num_vars

use std::marker::PhantomData;

use crate::{
    expression::{gather_combine_all_evals, Expression, ExpressionStandard},
    mle::MleRef,
    prover::SumcheckProof,
};
use remainder_shared_types::{transcript::Transcript, FieldExt};
use serde::{Deserialize, Serialize};

use super::{claims::ClaimError, layer_enum::LayerEnum, Claim, Layer, LayerError, LayerId};

///A Layer with 0 num_vars
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
pub struct EmptyLayer<F, Tr> {
    pub(crate) expr: ExpressionStandard<F>,
    id: LayerId,
    _marker: PhantomData<Tr>,
}

impl<F: FieldExt, Tr: Transcript<F>> Layer<F> for EmptyLayer<F, Tr> {
    type Transcript = Tr;

    fn prove_rounds(
        &mut self,
        _: super::Claim<F>,
        _: &mut Self::Transcript,
    ) -> Result<SumcheckProof<F>, LayerError> {
        let eval =
            gather_combine_all_evals(&self.expr).map_err(|err| LayerError::ExpressionError(err))?;

        Ok(vec![vec![eval]].into())
    }

    fn verify_rounds(
        &mut self,
        claim: super::Claim<F>,
        sumcheck_rounds: Vec<Vec<F>>,
        _: &mut Self::Transcript,
    ) -> Result<(), LayerError> {
        if sumcheck_rounds[0][0] != claim.1 {
            return Err(LayerError::VerificationError(
                super::VerificationError::GKRClaimCheckFailed,
            ));
        }

        Ok(())
    }

    fn get_enum(self) -> LayerEnum<F, Tr> {
        LayerEnum::EmptyLayer(self)
    }

    fn get_claims(&self) -> Result<Vec<(LayerId, Claim<F>)>, LayerError> {
        // First off, parse the expression that is associated with the layer...
        // Next, get to the actual claims that are generated by each expression and grab them
        // Return basically a list of (usize, Claim)
        // let layerwise_expr = self.expression();

        // --- Define how to parse the expression tree ---
        // - Basically we just want to go down it and pass up claims
        // - We can only add a new claim if we see an MLE with all its indices bound

        let mut claims: Vec<Claim<F>> = Vec::new();
        let mut indices: Vec<LayerId> = Vec::new();

        let mut observer_fn = |exp: &ExpressionStandard<F>| {
            match exp {
                ExpressionStandard::Mle(mle_ref) => {
                    // --- First ensure that all the indices are fixed ---
                    let mle_indices = mle_ref.mle_indices();

                    // --- This is super jank ---
                    let mut fixed_mle_indices: Vec<F> = vec![];
                    for mle_idx in mle_indices {
                        fixed_mle_indices.push(mle_idx.val().ok_or(ClaimError::MleRefMleError)?);
                    }

                    // --- Grab the layer ID (i.e. MLE index) which this mle_ref refers to ---
                    let mle_layer_id = mle_ref.get_layer_id();

                    // --- Grab the actual value that the claim is supposed to evaluate to ---
                    if mle_ref.bookkeeping_table().len() != 1 {
                        return Err(ClaimError::MleRefMleError);
                    }
                    let claimed_value = mle_ref.bookkeeping_table()[0];

                    // --- Construct the claim ---
                    let claim: Claim<F> = (fixed_mle_indices, claimed_value);

                    // --- Push it into the list of claims ---
                    // --- Also push the layer_id ---
                    claims.push(claim);
                    indices.push(mle_layer_id);
                }
                ExpressionStandard::Product(mle_refs) => {
                    for mle_ref in mle_refs {
                        // --- First ensure that all the indices are fixed ---
                        let mle_indices = mle_ref.mle_indices();

                        // --- This is super jank ---
                        let mut fixed_mle_indices: Vec<F> = vec![];
                        for mle_idx in mle_indices {
                            fixed_mle_indices
                                .push(mle_idx.val().ok_or(ClaimError::MleRefMleError)?);
                        }

                        // --- Grab the layer ID (i.e. MLE index) which this mle_ref refers to ---
                        let mle_layer_id = mle_ref.get_layer_id();

                        // --- Grab the actual value that the claim is supposed to evaluate to ---
                        if mle_ref.bookkeeping_table().len() != 1 {
                            return Err(ClaimError::MleRefMleError);
                        }
                        let claimed_value = mle_ref.bookkeeping_table()[0];

                        // --- Construct the claim ---
                        let claim: Claim<F> = (fixed_mle_indices, claimed_value);

                        // --- Push it into the list of claims ---
                        // --- Also push the layer_id ---
                        claims.push(claim);
                        indices.push(mle_layer_id);
                    }
                }
                _ => {}
            }
            Ok(())
        };

        // // TODO!(ryancao): What the heck is this code doing?
        self.expr
            .traverse(&mut observer_fn)
            .map_err(LayerError::ClaimError)?;

        Ok(indices.into_iter().zip(claims).collect())
    }

    fn id(&self) -> &LayerId {
        &self.id
    }

    fn get_wlx_evaluations(
        &self,
        claim_vecs: Vec<Vec<F>>,
        claimed_vals: &mut Vec<F>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, ClaimError> {
        unimplemented!()
    }

    fn new<L: super::LayerBuilder<F>>(builder: L, id: LayerId) -> Self
    where
        Self: Sized,
    {
        Self {
            id,
            expr: builder.build_expression(),
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt, Tr: Transcript<F>> EmptyLayer<F, Tr> {
    ///Gets this layer's underlying expression
    pub fn expression(&self) -> &ExpressionStandard<F> {
        &self.expr
    }

    pub(crate) fn new_raw(id: LayerId, expr: ExpressionStandard<F>) -> Self {
        Self {
            id,
            expr,
            _marker: PhantomData
        }
    }
}
