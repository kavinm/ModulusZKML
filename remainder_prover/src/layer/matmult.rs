// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use std::marker::{PhantomData, PhantomPinned};

use ::serde::{Deserialize, Serialize};
use ark_std::{cfg_into_iter, end_timer, log2, start_timer};
use itertools::Itertools;
use log::debug;
use ndarray::Array2;
use rand::Rng;
use remainder_shared_types::{transcript::Transcript, FieldExt};

use super::{
    claims::{Claim, ClaimError, ENABLE_PRE_FIX, ENABLE_RAW_MLE},
    combine_mle_refs::{combine_mle_refs_with_aggregate, pre_fix_mle_refs},
    layer_enum::LayerEnum,
    Layer, LayerBuilder, LayerError, LayerId,
};
use crate::{
    expression::ExpressionStandard,
    gate::gate_helpers::{check_fully_bound, compute_sumcheck_message_no_beta_table},
    layer::{claims::get_num_wlx_evaluations, VerificationError},
    mle::{
        beta::BetaTable,
        dense::{DenseMle, DenseMleRef},
        mle_enum::MleEnum,
        Mle, MleIndex, MleRef,
    },
    prover::SumcheckProof,
    sumcheck::{
        compute_sumcheck_message, evaluate_at_a_point, get_round_degree, Evals, VerifyError,
    },
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
pub struct Matrix<F: FieldExt> {
    pub mle_ref: DenseMleRef<F>,
    pub num_rows_vars: usize,
    pub num_cols_vars: usize,
    pub prefix_bits: Option<Vec<MleIndex<F>>>,
}

impl<F: FieldExt> Matrix<F> {
    /// For weight matrices
    pub fn new(
        mle_ref: DenseMleRef<F>,
        num_rows: usize,
        num_cols: usize,
        prefix_bits: Option<Vec<MleIndex<F>>>,
    ) -> Matrix<F> {
        Matrix {
            mle_ref,
            num_rows_vars: log2(num_rows) as usize,
            num_cols_vars: log2(num_cols) as usize,
            prefix_bits,
        }
    }

    /// For circuit intermediates
    pub fn new_with_padding(
        mut mle_ref: DenseMleRef<F>,
        num_rows: usize,
        num_cols: usize,
        prefix_bits: Option<Vec<MleIndex<F>>>,
    ) -> Matrix<F> {
        assert_eq!(mle_ref.bookkeeping_table().len(), num_rows * num_cols);

        // pad the columns
        if 1 << log2(num_cols) != num_cols {
            let mut new_bookkeeping_table = Vec::new();
            let num_to_pad_each_row = (1 << log2(num_cols) as usize) - num_cols;
            for chunk in mle_ref.bookkeeping_table.chunks(num_cols) {
                new_bookkeeping_table.extend(
                    [chunk.to_vec(), vec![F::zero(); num_to_pad_each_row]]
                        .into_iter()
                        .concat(),
                )
            }
            mle_ref.bookkeeping_table = new_bookkeeping_table;
        }

        // pad the rows
        let padded_matrix_len = (1 << log2(num_rows) as usize) * (1 << log2(num_cols) as usize);
        if mle_ref.bookkeeping_table.len() != padded_matrix_len {
            let num_need_to_pad = padded_matrix_len - mle_ref.bookkeeping_table.len();
            mle_ref.bookkeeping_table =
                [mle_ref.bookkeeping_table, vec![F::zero(); num_need_to_pad]]
                    .into_iter()
                    .concat()
        }

        assert_eq!(padded_matrix_len, mle_ref.bookkeeping_table.len());

        Matrix {
            mle_ref,
            num_rows_vars: log2(num_rows) as usize,
            num_cols_vars: log2(num_cols) as usize,
            prefix_bits,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
pub struct MatMult<F: FieldExt, Tr: Transcript<F>> {
    pub layer_id: LayerId,
    matrix_a: Matrix<F>,
    matrix_b: Matrix<F>,
    num_vars_middle_ab: Option<usize>,
    _marker: PhantomData<F>,
    _marker2: PhantomData<Tr>,
}

impl<F: FieldExt, Tr: Transcript<F>> MatMult<F, Tr> {
    pub fn new(layer_id: LayerId, matrix_a: Matrix<F>, matrix_b: Matrix<F>) -> MatMult<F, Tr> {
        MatMult {
            layer_id,
            matrix_a,
            matrix_b,
            num_vars_middle_ab: None,
            _marker: PhantomData,
            _marker2: PhantomData,
        }
    }

    fn pre_processing_step(&mut self, claim_a: Vec<F>, claim_b: Vec<F>) {
        let matrix_a_mle_ref = &mut self.matrix_a.mle_ref;
        let mut matrix_b_mle_ref = &mut self.matrix_b.mle_ref;

        // check that both matrices are padded
        assert_eq!(
            (1 << self.matrix_a.num_cols_vars) * (1 << self.matrix_a.num_rows_vars),
            matrix_a_mle_ref.bookkeeping_table().len()
        );
        assert_eq!(
            (1 << self.matrix_b.num_cols_vars) * (1 << self.matrix_b.num_rows_vars),
            matrix_b_mle_ref.bookkeeping_table().len()
        );

        // check to make sure the dimensions match
        if self.matrix_a.num_cols_vars == self.matrix_b.num_rows_vars {
            self.num_vars_middle_ab = Some(self.matrix_a.num_cols_vars);
        } else {
            // TODO: raise error
        }

        let transpose_timer = start_timer!(|| "transpose matrix");
        let mut matrix_a_transp = gen_transpose_matrix(
            &matrix_a_mle_ref,
            self.matrix_a.num_rows_vars,
            self.matrix_a.num_cols_vars,
            self.matrix_a.prefix_bits.clone(),
        );
        end_timer!(transpose_timer);

        matrix_a_transp.index_mle_indices(0);
        matrix_b_mle_ref.index_mle_indices(0);

        // bind the row indices of matrix a to relevant claim point
        claim_a.into_iter().enumerate().for_each(|(idx, chal)| {
            matrix_a_transp.fix_variable(idx, chal);
        });
        let mut bound_indices_a = vec![];

        let new_a_indices = matrix_a_transp
            .mle_indices
            .clone()
            .into_iter()
            .filter_map(|index: MleIndex<F>| {
                if let MleIndex::IndexedBit(_) = index {
                    Some(MleIndex::Iterated)
                } else if let MleIndex::Bound(..) = index {
                    bound_indices_a.push(index);
                    None
                } else {
                    Some(index)
                }
            })
            .collect_vec();

        self.matrix_a.mle_ref = DenseMle::new_from_raw(
            matrix_a_transp.bookkeeping_table().clone().to_vec(),
            matrix_a_transp.layer_id,
            None,
        )
        .mle_ref();
        self.matrix_a.mle_ref.mle_indices = new_a_indices
            .into_iter()
            .chain(bound_indices_a.into_iter())
            .collect_vec();
        self.matrix_a.mle_ref.index_mle_indices(0);

        // bind the column indices of matrix b to relevant claim point
        claim_b.into_iter().enumerate().for_each(|(idx, chal)| {
            matrix_b_mle_ref.fix_variable(idx, chal);
        });
        let new_b_indices = matrix_b_mle_ref
            .clone()
            .mle_indices
            .into_iter()
            .map(|index| {
                if let MleIndex::IndexedBit(_) = index {
                    MleIndex::Iterated
                } else {
                    index
                }
            })
            .collect_vec();
        matrix_b_mle_ref.mle_indices = new_b_indices;
        matrix_b_mle_ref.index_mle_indices(0);
    }

    /// dummy sumcheck prover for this, testing purposes
    fn dummy_prove_rounds(
        &mut self,
        claim: Claim<F>,
        rng: &mut impl Rng,
    ) -> Result<Vec<(Vec<F>, Option<F>)>, LayerError> {
        let mut claim_b = claim.get_point().clone();
        let claim_a = claim_b.split_off(self.matrix_b.num_cols_vars);

        self.pre_processing_step(claim_a, claim_b);

        let mut messages: Vec<(Vec<F>, Option<F>)> = vec![];
        let mut challenges: Vec<F> = vec![];
        let mut challenge: Option<F> = None;

        let first_message = compute_sumcheck_message_no_beta_table(
            &[self.matrix_a.mle_ref.clone(), self.matrix_b.mle_ref.clone()],
            0,
            2,
        )
        .unwrap();
        messages.push((first_message, challenge));

        let num_vars_middle = self.num_vars_middle_ab.unwrap(); // TODO: raise error if not

        for round in 1..num_vars_middle {
            // TODO: raise error if None
            challenge = Some(F::from(rng.gen::<u64>()));
            let chal = challenge.unwrap();
            challenges.push(chal);
            self.matrix_a
                .mle_ref
                .fix_variable(round - 1, challenge.clone().unwrap());
            self.matrix_b
                .mle_ref
                .fix_variable(round - 1, challenge.clone().unwrap());
            let next_message = compute_sumcheck_message_no_beta_table(
                &[self.matrix_a.mle_ref.clone(), self.matrix_b.mle_ref.clone()],
                round,
                2,
            )
            .unwrap();
            messages.push((next_message, challenge));
        }

        Ok(messages)
    }

    /// dummy verifier for dummy sumcheck, testing purposes
    fn dummy_verify_rounds(
        &mut self,
        messages: Vec<(Vec<F>, Option<F>)>,
        rng: &mut impl Rng,
        claim: Claim<F>,
    ) -> Result<(), VerifyError> {
        // first message check
        let mut prev_evals = &messages[0].0;
        let mut challenges = vec![];

        let mut claim_b = claim.get_point().clone();
        let claim_a = claim_b.split_off(self.matrix_b.num_cols_vars);

        let claimed_val = prev_evals[0] + prev_evals[1];
        if claimed_val != claim.get_result() {
            dbg!("hello");
            dbg!(messages[0].0[0] + messages[0].0[1]);
            return Err(VerifyError::SumcheckBad);
        }

        let num_vars_middle = self.num_vars_middle_ab.unwrap(); // TODO: raise error if not

        // --- Go through sumcheck messages + (FS-generated) challenges ---
        for i in 1..num_vars_middle {
            // TODO: raise error if not
            let (evals, challenge) = &messages[i];
            let curr_evals = evals;
            let chal = (challenge).unwrap();
            // --- Evaluate the previous round's polynomial at the random challenge point, i.e. g_{i - 1}(r_i) ---
            let prev_at_r = evaluate_at_a_point(prev_evals, challenge.unwrap())
                .expect("could not evaluate at challenge point");

            if prev_at_r != curr_evals[0] + curr_evals[1] {
                dbg!("whoops");
                dbg!(&prev_at_r);
                dbg!(curr_evals[0] + curr_evals[1]);
                return Err(VerifyError::SumcheckBad);
            };
            prev_evals = curr_evals;
            challenges.push(chal);
        }

        let final_chal = F::from(rng.gen::<u64>());
        challenges.push(final_chal);
        self.matrix_a
            .mle_ref
            .fix_variable(num_vars_middle - 1, final_chal);
        self.matrix_b
            .mle_ref
            .fix_variable(num_vars_middle - 1, final_chal);

        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).unwrap();

        let full_claim_chals_a = challenges
            .clone()
            .into_iter()
            .chain(claim_a.into_iter())
            .collect_vec();
        let full_claim_chals_b = claim_b
            .into_iter()
            .chain(challenges.into_iter())
            .collect_vec();

        let fully_bound_a =
            check_fully_bound(&mut [self.matrix_a.mle_ref.clone()], full_claim_chals_a).unwrap();
        let fully_bound_b =
            check_fully_bound(&mut [self.matrix_b.mle_ref.clone()], full_claim_chals_b).unwrap();
        let matrix_product = fully_bound_a * fully_bound_b;

        if prev_at_r != matrix_product {
            return Err(VerifyError::SumcheckBad);
        }

        Ok(())
    }
}

impl<F: FieldExt, Tr: Transcript<F>> Layer<F> for MatMult<F, Tr> {
    type Transcript = Tr;

    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript: &mut Self::Transcript,
    ) -> Result<SumcheckProof<F>, LayerError> {
        let mut claim_b = claim.get_point().clone();
        let claim_a = claim_b.split_off(self.matrix_b.num_cols_vars);

        let pre_process_timer = start_timer!(|| "start preprocessing step");
        self.pre_processing_step(claim_a, claim_b);
        end_timer!(pre_process_timer);

        let mut challenges: Vec<F> = vec![];

        let first_message = compute_sumcheck_message_no_beta_table(
            &[self.matrix_a.mle_ref.clone(), self.matrix_b.mle_ref.clone()],
            0,
            2,
        )
        .unwrap();
        transcript
            .append_field_elements("Initial Sumcheck evaluations", &first_message)
            .unwrap();

        let val = claim.get_result();
        if val != first_message[0] + first_message[1] {
            dbg!(&val);
            dbg!(first_message[0] + first_message[1]);
        }

        let num_vars_middle = self.num_vars_middle_ab.unwrap(); // TODO: raise error if not

        let sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(first_message))
            .chain((1..num_vars_middle).map(|round| {
                let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
                challenges.push(challenge);
                self.matrix_a.mle_ref.fix_variable(round - 1, challenge);
                self.matrix_b.mle_ref.fix_variable(round - 1, challenge);
                let next_message = compute_sumcheck_message_no_beta_table(
                    &[self.matrix_a.mle_ref.clone(), self.matrix_b.mle_ref.clone()],
                    round,
                    2,
                )
                .unwrap();

                transcript
                    .append_field_elements("Sumcheck evaluations", &next_message)
                    .unwrap();
                Ok::<_, LayerError>(next_message)
            }))
            .try_collect()?;

        let final_chal = transcript
            .get_challenge("Final Sumcheck challenge for binding x")
            .unwrap();
        challenges.push(final_chal);
        self.matrix_a
            .mle_ref
            .fix_variable(num_vars_middle - 1, final_chal);
        self.matrix_b
            .mle_ref
            .fix_variable(num_vars_middle - 1, final_chal);

        Ok(sumcheck_rounds.into())
    }

    /// Verifies the sumcheck protocol
    fn verify_rounds(
        &mut self,
        claim: Claim<F>,
        sumcheck_prover_messages: Vec<Vec<F>>,
        transcript: &mut Self::Transcript,
    ) -> Result<(), LayerError> {
        let mut challenges = vec![];

        let mut prev_evals = &sumcheck_prover_messages[0];
        let mut claim_b = claim.get_point().clone();
        let claim_a = claim_b.split_off(self.matrix_b.num_cols_vars);

        let claimed_val = prev_evals[0] + prev_evals[1];

        if claimed_val != claim.get_result() {
            debug!("I'm the PROBLEM");
            debug!("msg0 + msg1 =\n{:?}", prev_evals[0] + prev_evals[1]);
            debug!("rest =\n{:?}", claim.get_result());
            return Err(LayerError::VerificationError(
                VerificationError::SumcheckStartFailed,
            ));
        }

        transcript
            .append_field_elements("Initial Sumcheck evaluations", &sumcheck_prover_messages[0])
            .unwrap();

        // --- For round 1 < i < n, perform the check ---
        // g_{i - 1}(r_i) = g_i(0) + g_i(1)
        for curr_evals in sumcheck_prover_messages.iter().skip(1) {
            let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();

            let prev_at_r =
                evaluate_at_a_point(prev_evals, challenge).map_err(LayerError::InterpError)?;

            if prev_at_r != curr_evals[0] + curr_evals[1] {
                return Err(LayerError::VerificationError(
                    VerificationError::SumcheckFailed,
                ));
            };

            transcript
                .append_field_elements("Sumcheck evaluations", curr_evals)
                .unwrap();

            prev_evals = curr_evals;
            challenges.push(challenge);
        }

        // --- In the final round, we check that g(r_1, ..., r_n) = g_n(r_n) ---
        // Here, we first sample r_n.
        let final_chal = transcript
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        challenges.push(final_chal);

        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).unwrap();
        let full_claim_chals_a = challenges
            .clone()
            .into_iter()
            .chain(claim_a.into_iter())
            .collect_vec();
        let full_claim_chals_b = claim_b
            .into_iter()
            .chain(challenges.into_iter())
            .collect_vec();
        let fully_bound_a =
            check_fully_bound(&mut [self.matrix_a.mle_ref.clone()], full_claim_chals_a).unwrap();
        let fully_bound_b =
            check_fully_bound(&mut [self.matrix_b.mle_ref.clone()], full_claim_chals_b).unwrap();
        let matrix_product = fully_bound_a * fully_bound_b;

        if prev_at_r != matrix_product {
            return Err(LayerError::VerificationError(
                VerificationError::FinalSumcheckFailed,
            ));
        }

        Ok(())
    }

    /// Get the claims that this layer makes on other layers
    fn get_claims(&self) -> Result<Vec<Claim<F>>, LayerError> {
        let claims = vec![&self.matrix_a, &self.matrix_b]
            .into_iter()
            .map(|matrix| {
                let matrix_fixed_indices = matrix
                    .mle_ref
                    .mle_indices()
                    .into_iter()
                    .map(|index| {
                        index
                            .val()
                            .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))
                            .unwrap()
                    })
                    .collect_vec();

                let matrix_val = matrix.mle_ref.bookkeeping_table()[0];
                let claim: Claim<F> = Claim::new(
                    matrix_fixed_indices,
                    matrix_val,
                    Some(self.id().clone()),
                    Some(matrix.mle_ref.layer_id),
                    Some(MleEnum::Dense(matrix.mle_ref.clone())),
                );
                claim
            })
            .collect_vec();

        Ok(claims)
    }

    ///Gets this layers id
    fn id(&self) -> &LayerId {
        &self.layer_id
    }

    ///Create new ConcreteLayer from a LayerBuilder
    fn new<L: LayerBuilder<F>>(_builder: L, _id: LayerId) -> Self {
        todo!()
    }

    fn get_wlx_evaluations(
        &self,
        claim_vecs: &Vec<Vec<F>>,
        claimed_vals: &Vec<F>,
        claim_mle_refs: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, ClaimError> {
        let matrix_a_og: DenseMleRef<F> = DenseMle::new_from_raw(
            self.matrix_a.clone().mle_ref.original_bookkeeping_table,
            self.matrix_a.mle_ref.layer_id,
            None,
        )
        .mle_ref();

        let matrix_b_og: DenseMleRef<F> = DenseMle::new_from_raw(
            self.matrix_b.clone().mle_ref.original_bookkeeping_table,
            self.matrix_b.mle_ref.layer_id,
            None,
        )
        .mle_ref();

        let mut expr = ExpressionStandard::Product(vec![matrix_a_og, matrix_b_og]);

        // get the number of evaluations
        let (num_evals, common_idx) = get_num_wlx_evaluations(claim_vecs);

        let mut claim_mle_refs = claim_mle_refs.clone();

        if ENABLE_PRE_FIX {
            if common_idx.is_some() {
                pre_fix_mle_refs(&mut claim_mle_refs, &claim_vecs[0], common_idx.unwrap());
            }
        }

        let mut degree = 0;
        if !ENABLE_RAW_MLE {
            expr.index_mle_indices(0);
            degree = get_round_degree(&expr, 0);
        }

        // we already have the first #claims evaluations, get the next num_evals - #claims evaluations
        let next_evals: Vec<F> = cfg_into_iter!(num_claims..num_evals)
            .map(|idx| {
                // get the challenge l(idx)
                let new_chal: Vec<F> = cfg_into_iter!(0..num_idx)
                    .map(|claim_idx| {
                        let evals: Vec<F> = cfg_into_iter!(&claim_vecs)
                            .map(|claim| claim[claim_idx])
                            .collect();
                        evaluate_at_a_point(&evals, F::from(idx as u64)).unwrap()
                    })
                    .collect();

                if !ENABLE_RAW_MLE {
                    let mut beta = BetaTable::new(new_chal).unwrap();
                    beta.table.index_mle_indices(0);
                    let eval = compute_sumcheck_message(&expr, 0, degree, &beta).unwrap();
                    let Evals(evals) = eval;
                    evals[0] + evals[1]
                } else {
                    let wlx_eval_on_mle_ref =
                        combine_mle_refs_with_aggregate(&claim_mle_refs, &new_chal);
                    wlx_eval_on_mle_ref.unwrap()
                }
            })
            .collect();

        // concat this with the first k evaluations from the claims to
        // get num_evals evaluations
        let mut wlx_evals = claimed_vals.clone();
        wlx_evals.extend(&next_evals);
        Ok(wlx_evals)
    }

    fn get_enum(self) -> crate::layer::layer_enum::LayerEnum<F, Self::Transcript> {
        LayerEnum::MatMult(self)
    }
}

impl<F: std::fmt::Debug + FieldExt, Tr: Transcript<F>> MatMult<F, Tr> {
    pub(crate) fn circuit_description_fmt<'a>(&'a self) -> impl std::fmt::Display + 'a {
        // --- Dummy struct which simply exists to implement `std::fmt::Display` ---
        // --- so that it can be returned as an `impl std::fmt::Display` ---
        struct MatMultCircuitDesc<'a, F: std::fmt::Debug + FieldExt, Tr: Transcript<F>>(
            &'a MatMult<F, Tr>,
        );

        impl<'a, F: std::fmt::Debug + FieldExt, Tr: Transcript<F>> std::fmt::Display
            for MatMultCircuitDesc<'a, F, Tr>
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("MatMult")
                    .field("matrix_a_layer_id", &self.0.matrix_a.mle_ref.layer_id)
                    .field("matrix_a_mle_indices", &self.0.matrix_a.mle_ref.mle_indices)
                    .field("matrix_b_layer_id", &self.0.matrix_b.mle_ref.layer_id)
                    .field(
                        "matrix_b_mle_indices",
                        &self.0.matrix_b.mle_ref.mle_indices(),
                    )
                    .field("num_vars_middle_ab", &self.0.matrix_a.num_cols_vars)
                    .finish()
            }
        }
        MatMultCircuitDesc(self)
    }
}

pub fn gen_transpose_matrix<F: FieldExt>(
    matrix: &DenseMleRef<F>,
    num_rows_vars: usize,
    num_cols_vars: usize,
    prefix_bits: Option<Vec<MleIndex<F>>>,
) -> DenseMleRef<F> {
    let num_rows = 1 << num_rows_vars;
    let num_cols = 1 << num_cols_vars;

    let matrix_array_2 =
        Array2::from_shape_vec((num_rows, num_cols), matrix.bookkeeping_table.clone()).unwrap();
    let matrix_transpose = matrix_array_2.reversed_axes();
    let matrix_transp_vec = matrix_transpose
        .outer_iter()
        .map(|x| x.to_vec())
        .flat_map(|row| row)
        .collect_vec();
    DenseMle::new_from_raw(matrix_transp_vec, matrix.layer_id, prefix_bits).mle_ref()
}

pub fn product_two_matrices<F: FieldExt>(matrix_a: Matrix<F>, matrix_b: Matrix<F>) -> Vec<F> {
    let num_middle_ab = 1 << matrix_a.num_cols_vars;

    let matrix_b_transpose = gen_transpose_matrix(
        &matrix_b.mle_ref,
        matrix_b.num_rows_vars,
        matrix_b.num_cols_vars,
        matrix_a.prefix_bits,
    );
    let product_matrix = matrix_a
        .mle_ref
        .bookkeeping_table
        .chunks(num_middle_ab as usize)
        .flat_map(|chunk_a| {
            matrix_b_transpose
                .bookkeeping_table
                .chunks(num_middle_ab)
                .map(|chunk_b| {
                    chunk_a
                        .iter()
                        .zip(chunk_b.iter())
                        .fold(F::zero(), |acc, (&a, &b)| acc + (a * b))
                })
                .collect_vec()
        })
        .collect_vec();

    product_matrix
}

#[cfg(test)]
mod test {
    use ark_std::{log2, test_rng};
    use remainder_shared_types::{transcript::poseidon_transcript::PoseidonTranscript, Fr};

    use crate::{
        layer::{
            claims::Claim,
            matmult::{gen_transpose_matrix, product_two_matrices, Matrix},
            LayerId,
        },
        mle::{
            dense::{DenseMle, DenseMleRef},
            MleRef,
        },
    };

    use super::MatMult;

    #[test]
    fn test_product_irregular_matrices() {
        let mle_vec_a = vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(9),
            Fr::from(10),
            Fr::from(13),
            Fr::from(1),
            Fr::from(3),
            Fr::from(10),
            Fr::from(2),
            Fr::from(9),
            Fr::from(10),
            Fr::from(1),
            Fr::from(3),
            Fr::from(10),
            Fr::from(2),
        ];
        let mle_vec_b = vec![
            Fr::from(3),
            Fr::from(5),
            Fr::from(9),
            Fr::from(6),
            Fr::from(5),
            Fr::from(9),
            Fr::from(6),
            Fr::from(1),
            Fr::from(3),
        ];

        let mle_a: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_a, LayerId::Input(0), None);
        let mle_b: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_b, LayerId::Input(0), None);

        let matrix_a = Matrix::new(mle_a.mle_ref(), 5, 3, None);
        let matrix_b = Matrix::new(mle_b.mle_ref(), 3, 3, None);

        let res_product = product_two_matrices(matrix_a, matrix_b);

        // 1  2  9
        // 10 13 1       3  5  9
        // 3  10 2   `   6  5  9
        // 9  10 1       6  1  3
        // 3  10 2

        let exp_product = vec![
            Fr::from(1 * 3 + 2 * 6 + 9 * 6),
            Fr::from(1 * 5 + 2 * 5 + 9 * 1),
            Fr::from(1 * 9 + 2 * 9 + 9 * 3),
            Fr::from(10 * 3 + 13 * 6 + 1 * 6),
            Fr::from(10 * 5 + 13 * 5 + 1 * 1),
            Fr::from(10 * 9 + 13 * 9 + 1 * 3),
            Fr::from(3 * 3 + 10 * 6 + 2 * 6),
            Fr::from(3 * 5 + 10 * 5 + 2 * 1),
            Fr::from(3 * 9 + 10 * 9 + 2 * 3),
            Fr::from(9 * 3 + 10 * 6 + 1 * 6),
            Fr::from(9 * 5 + 10 * 5 + 1 * 1),
            Fr::from(9 * 9 + 10 * 9 + 1 * 3),
            Fr::from(3 * 3 + 10 * 6 + 2 * 6),
            Fr::from(3 * 5 + 10 * 5 + 2 * 1),
            Fr::from(3 * 9 + 10 * 9 + 2 * 3),
        ];

        let mle_out: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(exp_product, LayerId::Input(0), None);
        let matrix_out = Matrix::new(mle_out.mle_ref(), 5, 3, None);

        assert_eq!(res_product, matrix_out.mle_ref.bookkeeping_table());
    }

    #[test]
    fn test_sumcheck_irregular_matrices() {
        let mut rng = test_rng();

        let mle_vec_a = vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(9),
            Fr::from(10),
            Fr::from(13),
            Fr::from(1),
            Fr::from(3),
            Fr::from(10),
            Fr::from(2),
            Fr::from(9),
            Fr::from(10),
            Fr::from(1),
            Fr::from(3),
            Fr::from(10),
            Fr::from(2),
        ];
        let mle_vec_b = vec![
            Fr::from(3),
            Fr::from(5),
            Fr::from(9),
            Fr::from(6),
            Fr::from(5),
            Fr::from(9),
            Fr::from(6),
            Fr::from(1),
            Fr::from(3),
        ];

        let mle_a: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_a, LayerId::Input(0), None);
        let mle_b: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_b, LayerId::Input(0), None);

        let matrix_a = Matrix::new(mle_a.mle_ref(), 5, 3, None);
        let matrix_b = Matrix::new(mle_b.mle_ref(), 3, 3, None);

        let res_product = product_two_matrices(matrix_a.clone(), matrix_b.clone());
        let mut mle_product_ref: DenseMleRef<Fr> =
            DenseMle::new_from_raw(res_product, LayerId::Input(0), None).mle_ref();

        let _ = mle_product_ref.index_mle_indices(0);

        mle_product_ref.fix_variable(0, Fr::from(1));
        mle_product_ref.fix_variable(1, Fr::from(2));
        mle_product_ref.fix_variable(2, Fr::from(3));
        mle_product_ref.fix_variable(3, Fr::from(4));
        mle_product_ref.fix_variable(4, Fr::from(5));

        assert_eq!(mle_product_ref.bookkeeping_table().len(), 1);

        let claim = Claim::new_raw(
            vec![
                Fr::from(1),
                Fr::from(2),
                Fr::from(3),
                Fr::from(4),
                Fr::from(5),
            ],
            mle_product_ref.bookkeeping_table()[0],
        );

        let mut matrix_init: MatMult<Fr, PoseidonTranscript<Fr>> =
            MatMult::new(LayerId::Input(0), matrix_a, matrix_b);

        let messages = matrix_init.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res = matrix_init.dummy_verify_rounds(messages.unwrap(), &mut rng, claim);

        assert!(verify_res.is_ok());
    }

    #[test]
    fn test_product_two_matrices() {
        let mle_vec_a = vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(9),
            Fr::from(10),
            Fr::from(13),
            Fr::from(1),
            Fr::from(3),
            Fr::from(10),
        ];
        let mle_vec_b = vec![Fr::from(3), Fr::from(5), Fr::from(9), Fr::from(6)];
        let mle_a: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_a, LayerId::Input(0), None);
        let mle_b: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_b, LayerId::Input(0), None);

        let matrix_a = Matrix::new(mle_a.mle_ref(), 4, 2, None);
        let matrix_b = Matrix::new(mle_b.mle_ref(), 2, 2, None);

        let res_product = product_two_matrices(matrix_a, matrix_b);

        let exp_product = vec![
            Fr::from(1 * 3 + 2 * 9),
            Fr::from(1 * 5 + 2 * 6),
            Fr::from(9 * 3 + 10 * 9),
            Fr::from(9 * 5 + 10 * 6),
            Fr::from(13 * 3 + 1 * 9),
            Fr::from(13 * 5 + 1 * 6),
            Fr::from(3 * 3 + 10 * 9),
            Fr::from(3 * 5 + 10 * 6),
        ];

        assert_eq!(res_product, exp_product);
    }

    #[test]
    fn test_product_two_matrices_2() {
        let mle_vec_a = vec![
            Fr::from(3),
            Fr::from(4),
            Fr::from(1),
            Fr::from(6),
            Fr::from(2),
            Fr::from(9),
            Fr::from(0),
            Fr::from(1),
            Fr::from(4),
            Fr::from(5),
            Fr::from(4),
            Fr::from(2),
            Fr::from(4),
            Fr::from(2),
            Fr::from(6),
            Fr::from(7),
            Fr::from(3),
            Fr::from(4),
            Fr::from(1),
            Fr::from(6),
            Fr::from(2),
            Fr::from(9),
            Fr::from(0),
            Fr::from(1),
            Fr::from(4),
            Fr::from(5),
            Fr::from(4),
            Fr::from(2),
            Fr::from(4),
            Fr::from(2),
            Fr::from(6),
            Fr::from(7),
        ];
        let mle_vec_b = vec![
            Fr::from(3),
            Fr::from(2),
            Fr::from(1),
            Fr::from(5),
            Fr::from(3),
            Fr::from(6),
            Fr::from(7),
            Fr::from(4),
        ];
        let mle_a: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_a, LayerId::Input(0), None);
        let mle_b: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_b, LayerId::Input(0), None);

        let matrix_a = Matrix::new(mle_a.mle_ref(), 8, 4, None);
        let matrix_b = Matrix::new(mle_b.mle_ref(), 4, 2, None);

        let res_product = product_two_matrices(matrix_a, matrix_b);

        let exp_product = vec![
            Fr::from(58),
            Fr::from(56),
            Fr::from(22),
            Fr::from(53),
            Fr::from(43),
            Fr::from(65),
            Fr::from(81),
            Fr::from(82),
            Fr::from(58),
            Fr::from(56),
            Fr::from(22),
            Fr::from(53),
            Fr::from(43),
            Fr::from(65),
            Fr::from(81),
            Fr::from(82),
        ];

        assert_eq!(res_product, exp_product);
    }

    #[test]
    fn test_transpose_code() {
        let mle_vec = vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(9),
            Fr::from(10),
            Fr::from(13),
            Fr::from(1),
            Fr::from(3),
            Fr::from(10),
        ];

        let mle_ref: DenseMleRef<Fr> =
            DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None).mle_ref();

        let expected_vec = vec![
            Fr::from(1),
            Fr::from(9),
            Fr::from(13),
            Fr::from(3),
            Fr::from(2),
            Fr::from(10),
            Fr::from(1),
            Fr::from(10),
        ];
        let expected_ref: DenseMleRef<Fr> =
            DenseMle::new_from_raw(expected_vec, LayerId::Input(0), None).mle_ref();

        assert_eq!(
            gen_transpose_matrix(&mle_ref, 2, 1, None).bookkeeping_table,
            expected_ref.bookkeeping_table
        );
    }

    #[test]
    /// super basic symmetric test
    fn test_sumcheck_1() {
        let mut rng = test_rng();
        let claim = Claim::new_raw(vec![Fr::from(1), Fr::from(0)], Fr::from(3));

        let matrix_a_vec = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)];
        let matrix_b_vec = vec![Fr::from(1), Fr::from(1), Fr::from(1), Fr::from(1)];
        let matrix_a_mle: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(matrix_a_vec, LayerId::Input(0), None);
        let matrix_b_mle: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(matrix_b_vec, LayerId::Input(0), None);

        let matrix_a: Matrix<Fr> = Matrix::new(matrix_a_mle.mle_ref(), 2, 2, None);
        let matrix_b: Matrix<Fr> = Matrix::new(matrix_b_mle.mle_ref(), 2, 2, None);

        let mut matrix_init: MatMult<Fr, PoseidonTranscript<Fr>> =
            MatMult::new(LayerId::Input(0), matrix_a, matrix_b);

        let messages = matrix_init.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res = matrix_init.dummy_verify_rounds(messages.unwrap(), &mut rng, claim);

        assert!(verify_res.is_ok());
    }

    #[test]
    /// super basic asymmetric test
    fn test_sumcheck_asymmetric() {
        let mut rng = test_rng();
        let claim = Claim::new_raw(vec![Fr::from(1)], Fr::from(8));

        let matrix_a_vec = vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(1),
            Fr::from(3),
            Fr::from(1),
            Fr::from(1),
            Fr::from(2),
        ];
        let matrix_b_vec = vec![Fr::from(1), Fr::from(2), Fr::from(1), Fr::from(1)];
        let matrix_a_mle: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(matrix_a_vec, LayerId::Input(0), None);
        let matrix_b_mle: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(matrix_b_vec, LayerId::Input(0), None);

        let matrix_a: Matrix<Fr> = Matrix::new(matrix_a_mle.mle_ref(), 2, 4, None);
        let matrix_b: Matrix<Fr> = Matrix::new(matrix_b_mle.mle_ref(), 4, 1, None);

        let mut matrix_init: MatMult<Fr, PoseidonTranscript<Fr>> =
            MatMult::new(LayerId::Input(0), matrix_a, matrix_b);

        let messages = matrix_init.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res = matrix_init.dummy_verify_rounds(messages.unwrap(), &mut rng, claim);

        assert!(verify_res.is_ok());
    }
}
