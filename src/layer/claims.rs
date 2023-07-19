use crate::{
    expression::{Expression, ExpressionStandard},
    mle::MleIndex,
    FieldExt,
};

// use itertools::Itertools;
use crate::mle::MleRef;
use crate::sumcheck::*;

use ark_std::{cfg_iter, cfg_into_iter};
use itertools::{izip, multizip, Itertools};
use thiserror::Error;
use rayon::{prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
}, slice::ParallelSlice};

use super::{Claim, Layer};

#[derive(Error, Debug, Clone)]
enum LayerError {
    #[error("The Layer has not finished the sumcheck protocol")]
    SumCheckNotComplete,
    #[error("MLE indices must all be fixed")]
    ClaimMleIndexError,
    #[error("Layer ID not assigned")]
    LayerMleError,
    #[error("MLE within MleRef has multiple values within it")]
    MleRefMleError,
    #[error("Error aggregating claims")]
    ClaimAggroError,
    #[error("Should be evaluating to a sum")]
    ExpressionEvalError,
}

///Take in a layer that has completed the sumcheck protocol and return a list of claims on the next layer
fn get_claims<F: FieldExt>(layer: &impl Layer<F>) -> Result<Vec<(usize, Claim<F>)>, LayerError> {
    // First off, parse the expression that is associated with the layer...
    // Next, get to the actual claims that are generated by each expression and grab them
    // Return basically a list of (usize, Claim)
    let layerwise_expr = layer.get_expression();

    // --- Define how to parse the expression tree ---
    // - Basically we just want to go down it and pass up claims
    // - We can only add a new claim if we see an MLE with all its indices bound

    let mut claims: Vec<Claim<F>> = Vec::new();
    let mut indices: Vec<usize> = Vec::new();

    let mut observer_fn = |exp: &ExpressionStandard<F>| {
        match exp {
            ExpressionStandard::Mle(mle_ref) => {
                // --- First ensure that all the indices are fixed ---
                let mle_indices = mle_ref.get_mle_indices();

                // --- This is super jank ---
                let mut fixed_mle_indices: Vec<F> = vec![];
                for mle_idx in mle_indices {
                    match mle_idx {
                        // --- We can't have either iterated or indexed bits ---
                        MleIndex::IndexedBit(_) | MleIndex::Iterated => {
                            return Err(LayerError::MleRefMleError);
                        }
                        // --- We can't have either iterated or indexed bits ---
                        MleIndex::Bound(idx) => {
                            fixed_mle_indices.push(*idx);
                        }
                        MleIndex::Fixed(one) => {
                            let idx = if *one { F::from(1_u64) } else { F::from(1_u64) };
                            fixed_mle_indices.push(idx);
                        }
                    }
                }

                // --- Grab the layer ID (i.e. MLE index) which this mle_ref refers to ---
                let mle_layer_id = match mle_ref.get_layer_id() {
                    None => {
                        return Err(LayerError::LayerMleError);
                    }
                    Some(layer_id) => layer_id,
                };

                // --- Grab the actual value that the claim is supposed to evaluate to ---
                if mle_ref.mle().len() != 1 {
                    return Err(LayerError::MleRefMleError);
                }
                // TODO(ryancao): Does this accidentally take ownership of that element?
                let claimed_value = mle_ref.mle()[0];

                // --- Construct the claim ---
                let claim: Claim<F> = (fixed_mle_indices, claimed_value);

                // --- Push it into the list of claims ---
                // --- Also push the layer_id ---
                claims.push(claim);
                indices.push(mle_layer_id);
            }
            _ => {}
        }
        Ok(())
    };

    let _result = layerwise_expr.traverse(&mut observer_fn);
    if let result = Err(LayerError::ClaimMleIndexError) {
        return result;
    }

    Ok(indices.into_iter().zip(claims).collect())
}

fn compute_lx<F: FieldExt>(
    expr: &mut ExpressionStandard<F>,
    challenge: &Vec<F>,
    num_claims: usize,
) -> Result<Vec<F>, LayerError> {
    //fix variable hella times
    //evaluate expr on the mutated expr

    let num_vars = expr.index_mle_indices(0);
    let num_evals = num_vars*num_claims;
    let lx_evals: Result<Vec<F>, LayerError> = cfg_into_iter!(0..num_evals)
        .map(
            |_idx| {
                let fix_expr = expr.clone();
                let mut fixed_expr = challenge.iter()
                    .enumerate()
                    .fold(
                        fix_expr,
                        |mut expr, (idx, chal_point)| {
                            expr.fix_variable(idx, *chal_point); 
                            expr
                        });
                let val = evaluate_expr(&mut fixed_expr, 0, 0).unwrap();
                match val {
                    SumOrEvals::Sum(evaluation) => Ok(evaluation),
                    SumOrEvals::Evals(_) => Err(LayerError::ExpressionEvalError)
                }
            }
        ).collect();
    lx_evals
}

/// Aggregate several claims into one
fn aggregate_claims<F: FieldExt>(
    claims: Vec<Claim<F>>,
    expr: &mut ExpressionStandard<F>,
    rchal: F,
) -> Claim<F> {
   
    let claim_vecs: Vec<_> = cfg_into_iter!(claims.clone()).map(|(claimidx, _)| claimidx).collect();
    let numidx = claim_vecs[0].len();

    let rstar: Vec<F> = cfg_into_iter!(0..numidx).map(
        |idx| {
            let evals: Vec<F> = claim_vecs.iter().map(|claim| claim[idx]).collect();
            evaluate_at_a_point(evals, rchal).unwrap()
        }
    ).collect();

    let lx = compute_lx(expr, &rstar, claims.len()).unwrap();

    let claimed_val = evaluate_at_a_point(lx, rchal);

    (rstar, claimed_val.unwrap())
}

mod test {

    use crate::mle::{dense::DenseMle, Mle};

    use super::*;
    use ark_bn254::Fr;
    use ark_std::One;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_get_claim() {
        // [1, 1, 1, 1] \oplus (1 - (1 * (1 + V[1, 1, 1, 1]))) * 2
        let expression1: ExpressionStandard<Fr> = ExpressionStandard::Constant(Fr::one());
        let mle = DenseMle::<_, Fr>::new(vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()]);
        let expression3 = ExpressionStandard::Mle(mle.mle_ref());
        let expression = expression1.clone() + expression3.clone();
        // let expression = expression1.clone() * expression;
        let expression = expression1 - expression;
        let expression = expression * Fr::from(2);
        let _expression = expression3.concat(expression);

        // TODO(ryancao): Need to create a layer and fix all the MLE variables...
    }

    #[test]
    fn test_aggro_claim() {
        let mle_v1 = vec![Fr::from(1), Fr::from(0), Fr::from(2), Fr::from(3)];
        let mle1: DenseMle<Fr, Fr> = DenseMle::new(mle_v1);
        let mle_ref = mle1.mle_ref();
        let mut expr = ExpressionStandard::Mle(mle_ref);
        let mut expr_copy = expr.clone();

        let claim1: Claim<Fr> = (vec![Fr::from(2), Fr::from(3)], Fr::from(14));
        let claim2: Claim<Fr> = (vec![Fr::one(), Fr::from(7)], Fr::from(21));

        let res: Claim<Fr> = aggregate_claims(vec![claim1, claim2], &mut expr, Fr::from(10));

        let fix_vars = vec![Fr::from(-8), Fr::from(43)];
        expr_copy.index_mle_indices(0);
        for i in 0..2 {
            expr_copy.fix_variable(i, fix_vars[i]);
        }
        let expr_eval = evaluate_expr(&mut expr_copy, 0, 0).unwrap();
        if let SumOrEvals::Sum(num) = expr_eval {
            let exp: Claim<Fr> = (fix_vars, num);
            assert_eq!(res, exp);
        }
    }
}
