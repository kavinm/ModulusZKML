use crate::{
    expression::{Expression, ExpressionStandard},
    mle::MleIndex,
    FieldExt,
};

// use itertools::Itertools;
use crate::mle::MleRef;

use thiserror::Error;

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
}

///Take in a layer that has completed the sumcheck protocol and return a list of claims on the next layer
fn get_claims<F: FieldExt>(
    layer: &impl Layer<F>,
) -> Result<Vec<(usize, Claim<F>)>, LayerError> {
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
                            let idx = if *one {
                                F::from(1_u64)
                            } else {
                                F::from(0_u64)
                            };
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
                if mle_ref.bookkeeping_table().len() != 1 {
                    return Err(LayerError::MleRefMleError);
                }
                // TODO(ryancao): Does this accidentally take ownership of that element?
                let claimed_value = mle_ref.bookkeeping_table()[0];

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

mod test {

    use crate::mle::{dense::DenseMle, Mle};

    use super::*;
    use ark_bn254::Fr;
    use ark_std::One;

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
}
