// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use crate::adapter::{convert_lcpc_to_halo, LigeroClaim, LigeroProof};
use crate::ligero_ml_helper::{get_ml_inner_outer_tensors, naive_eval_mle_at_challenge_point};
use crate::ligero_structs::{LigeroCommit, LigeroEncoding, LigeroEvalProof};
use crate::utils::get_ligero_matrix_dims;
use crate::{verify, LcProofAuxiliaryInfo, LcRoot};
use ark_std::log2;
use remainder_shared_types::{
    transcript::{Transcript as RemainderTranscript, TranscriptError},
    FieldExt,
};
use tracing::instrument;

use super::poseidon_ligero::PoseidonSpongeHasher;
use super::{commit, prove};

/// Computes and optionally serializes a multilinear Ligero commitment proof
///
/// TODO!(ryancao): Better error-handling
///
/// ## Arguments
///
/// * `coeffs` - Vector of coefficients from the original MLE. Length must be a power of 2!
/// * `log_num_rows` - Log base 2 of the number of Ligero matrix rows.
/// * `log_orig_num_cols` - Log base 2 of the number of Ligero matrix columns. Note that
///     this plus `log_num_rows` must be equivalent to `log2(coeffs.len())`
/// * `rho_inv` - Inverse of the code rate `rho`
/// * `ligero_root_filename` - Filename of the Ligero Merkle root to be saved to. Additionally,
///                            whether to also serialize the proof (using ark-serialize) into a file.
///
/// ## Returns
/// * `ligero_encoding` - Not useful; can be reconstructed from the eval proof
/// * `ligero_commit` - Commitment including encoded matrix
/// * `ligero_root` - Merkle tree root
/// * `aux` - Auxiliary info for Ligero
///
/// ## Examples
/// ```
/// // TODO!(ryancao)
/// ```
#[instrument(skip_all, level = "debug")]
pub fn poseidon_ml_commit_prove<F: FieldExt>(
    coeffs: &Vec<F>,
    log_num_rows: usize,
    log_orig_num_cols: usize,
    rho_inv: u8,
    _maybe_ligero_root_filename: Option<&str>,
) -> (
    LigeroEncoding<F>,
    crate::LcCommit<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>,
    LcRoot<LigeroEncoding<F>, F>,
    LcProofAuxiliaryInfo,
) {
    // --- Auxiliaries ---
    // let rho = 1. / (rho_inv as f64);
    let num_rows = 1 << log_num_rows;
    let orig_num_cols = 1 << log_orig_num_cols;
    let encoded_num_cols = orig_num_cols * (rho_inv as usize);

    // --- Sanitycheck ---
    assert!(coeffs.len().is_power_of_two());
    assert_eq!(coeffs.len(), num_rows * orig_num_cols);

    // --- Create commitment ---
    let enc = LigeroEncoding::<F>::new_from_dims(orig_num_cols, encoded_num_cols);
    let comm = commit::<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>(coeffs, &enc).unwrap();

    // --- Only component of commitment which needs to be sent to the verifier is the commitment root ---
    let root: LcRoot<LigeroEncoding<F>, F> = comm.get_root();

    // --- Serialize the commitment root ---
    // if let Some(ligero_root_filename) = maybe_ligero_root_filename {
    //     let mut encoded_root_bytes: Vec<u8> = Vec::new();
    //     let lc_root_to_be_encoded = LcRoot::<LigeroEncoding<F>, F> {
    //         root: root.root,
    //         _p: PhantomData,
    //     };
    //     let _ = lc_root_to_be_encoded
    //         .serialize_compressed(&mut encoded_root_bytes)
    //         .unwrap();
    //     fs::write(ligero_root_filename, encoded_root_bytes.clone())
    //         .expect("Unable to write root to file");
    // }

    let aux = LcProofAuxiliaryInfo {
        rho_inv,
        encoded_num_cols,
        orig_num_cols,
        num_rows,
    };

    // --- Return the auxiliaries + commitment ---
    (enc, comm, root, aux)
}

/// Computes and optionally serializes a multilinear Ligero evaluation proof
///
/// ## Arguments
///
/// * `coeffs` - Vector of coefficients from the original MLE. Length must be a power of 2!
/// * `rho_inv` - Inverse of the code rate `rho`
/// * `log_num_rows` - Log base 2 of the number of Ligero matrix rows.
/// * `log_orig_num_cols` - Log base 2 of the number of Ligero matrix columns. Note that
///     this plus `log_num_rows` must be equivalent to `log2(coeffs.len())`
/// * `challenge_coord` - Vector of challenge values, i.e. the evaluation point.
/// * `transcript` - The Poseidon Transcript to be used for Fiat-Shamir.
/// * `ligero_proof_filename` - The file to save the serialized evaluation proof to.
/// * `ligero_aux_data_filename` - The file to save the serialized auxiliary data to.
/// * `comm` - The actual Ligero commitment, as generated by `poseidon_ml_commit_prove`
/// * `root` - The Ligero root, as generated by `poseidon_ml_commit_prove`
///
/// ## Examples
/// ```
/// // TODO!(ryancao)
/// ```
pub fn poseidon_ml_eval_prove<F: FieldExt, T: RemainderTranscript<F>>(
    coeffs: &Vec<F>,
    rho_inv: u8,
    log_num_rows: usize,
    log_orig_num_cols: usize,
    challenge_coord: &Vec<F>,
    transcript: &mut T,
    _maybe_ligero_proof_filename: Option<&str>,
    _maybe_ligero_aux_data_filename: Option<&str>,
    _maybe_ligero_claim_filename: Option<&str>,
    comm: LigeroCommit<PoseidonSpongeHasher<F>, F>,
    root: LcRoot<LigeroEncoding<F>, F>,
) -> Result<
    (
        F,
        LigeroEvalProof<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>,
    ),
    TranscriptError,
> {
    // --- Auxiliaries ---
    let rho = 1. / (rho_inv as f64);
    let num_rows = 1 << log_num_rows;
    let orig_num_cols = 1 << log_orig_num_cols;

    // --- Sanitycheck ---
    assert_eq!(coeffs.len(), num_rows * orig_num_cols);

    // --- Compute "a" and "b" from `challenge_coord` ---
    let (_, outer_tensor) = get_ml_inner_outer_tensors(challenge_coord, num_rows, orig_num_cols);

    // --- Generate the transcript and write to it ---
    // --- Transcript includes the Merkle root, the code rate, and the number of columns to be sampled ---
    transcript.append_field_element("polycommit", root.root)?;

    // Tl;dr this gives us the random vectors to check well-formedness from
    // As well as the actual columns we're opening at, plus proofs that those
    // columns are consistent against the Merkle root
    let ratio = orig_num_cols as f64 / num_rows as f64;
    let enc = LigeroEncoding::<F>::new(coeffs.len(), rho, ratio); // This is basically just a wrapper
    let pf_ya: Result<LigeroEvalProof<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>, _> =
        prove(&comm, &outer_tensor[..], &enc, transcript);

    let pf = pf_ya.unwrap();

    // --- Return the evaluation point value ---
    // TODO!(ryancao): Do we need this?
    let eval = naive_eval_mle_at_challenge_point(&comm.coeffs, challenge_coord);

    let _claim = LigeroClaim {
        point: challenge_coord.clone(),
        eval,
    };

    // ------------------- SERIALIZATION -------------------

    // --- First the auxiliaries ---
    // if let Some(ligero_aux_data_filename) = maybe_ligero_aux_data_filename {
    //     let ligero_proof_aux_info = LcProofAuxiliaryInfo {
    //         rho_inv: rho_inv as u8,
    //         encoded_num_cols: comm.encoded_num_cols,
    //         orig_num_cols: comm.orig_num_cols,
    //         num_rows: comm.n_rows,
    //     };
    //     let mut encoded_proof_aux_info_bytes: Vec<u8> = Vec::new();
    //     let _ = ligero_proof_aux_info.serialize_compressed(&mut encoded_proof_aux_info_bytes);

    //     fs::write(ligero_aux_data_filename, encoded_proof_aux_info_bytes)
    //         .expect("Unable to write proof auxiliaries to file");
    // }

    // --- The proof as well ---
    // if let Some(ligero_proof_filename) = maybe_ligero_proof_filename {
    //     let mut encoded_proof_bytes: Vec<u8> = Vec::new();
    //     pf.serialize_compressed(&mut encoded_proof_bytes).unwrap();

    //     fs::write(ligero_proof_filename, encoded_proof_bytes.clone())
    //         .expect("Unable to write proof to file");
    // }

    // ------------------- END SERIALIZATION -------------------

    // --- Return the evaluation point value ---
    // TODO!(ryancao): Do we need this?
    let eval = naive_eval_mle_at_challenge_point(&comm.coeffs, challenge_coord);
    Ok((eval, pf))
}

/// API for Remainder's Ligero commitment. Note that this function automatically
/// picks the Ligero matrix size!
///
/// ## Arguments
///
/// * `input_mle_bookkeeping_table` - bookkeeping table for the combined input MLE
/// * `rho_inv` - The Ligero code rate
///
/// ## Returns
///
/// * `ligero_encoding` - Not useful; can be reconstructed from the eval proof
/// * `ligero_commit` - Commitment including encoded matrix
/// * `ligero_root` - Merkle tree root
/// * `aux` - Auxiliary info for Ligero
#[instrument(skip_all, level = "debug")]
pub fn remainder_ligero_commit_prove<F: FieldExt>(
    input_mle_bookkeeping_table: &Vec<F>,
    rho_inv: u8,
    ratio: f64,
) -> (
    LigeroEncoding<F>,
    crate::LcCommit<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>,
    LcRoot<LigeroEncoding<F>, F>,
    LcProofAuxiliaryInfo,
) {
    // --- Sanitycheck ---
    assert!(input_mle_bookkeeping_table.len().is_power_of_two());

    // --- Get Ligero matrix dims + sanitycheck ---
    let (num_rows, orig_num_cols, _) =
        get_ligero_matrix_dims(input_mle_bookkeeping_table.len(), rho_inv, ratio).unwrap();

    // --- NOTE: These will have to be passed in later for the eval proof phase ---
    poseidon_ml_commit_prove(
        input_mle_bookkeeping_table,
        log2(num_rows) as usize,
        log2(orig_num_cols) as usize,
        rho_inv,
        None,
    )
}

/// API for Remainder's Ligero eval proof. Note that this function
/// outputs a proof of type `FieldExt` rather than `FieldExt`!
///
/// ## Arguments
///
/// * `input_layer_bookkeeping_table` - The bookkeeping table for the combined
///     input MLE.
/// * `challenge_coord` - The challenge at which to open the input MLE.
/// * `transcript` - The FS transcript so far
/// * `aux` - Auxiliary Ligero commit info; should be generated by the `commit` fn
/// * `comm` - Ligero commitment (coeff matrix + aux); should be generated by the `commit` fn
/// * `root` - Actual Merkle root commitment; should be generated by the `commit` fn
///
/// ## Returns
/// * `h2_ligero_proof` - Halo2-compatible Ligero proof for the evaluation of the original
///     polynomial (as given in `comm`) at `challenge_coord`
pub fn remainder_ligero_eval_prove<F: FieldExt, T: RemainderTranscript<F>>(
    input_layer_bookkeeping_table: &Vec<F>,
    challenge_coord: &Vec<F>,
    transcript: &mut T,
    aux: LcProofAuxiliaryInfo,
    comm: LigeroCommit<PoseidonSpongeHasher<F>, F>,
    root: LcRoot<LigeroEncoding<F>, F>,
) -> LigeroProof<F> {
    // --- Sanitycheck ---
    assert!(input_layer_bookkeeping_table.len().is_power_of_two());

    // --- Extract data from aux ---
    let rho_inv = aux.rho_inv;
    let log_num_rows = log2(aux.num_rows) as usize;
    let log_orig_num_cols = log2(aux.orig_num_cols) as usize;

    let (_, proof) = poseidon_ml_eval_prove(
        input_layer_bookkeeping_table,
        rho_inv,
        log_num_rows,
        log_orig_num_cols,
        challenge_coord,
        transcript,
        None,
        None,
        None,
        comm,
        root.clone(),
    )
    .unwrap();

    convert_lcpc_to_halo(aux, root, proof)
}

/// Function for verification of Ligero proof
pub fn remainder_ligero_verify<F: FieldExt>(
    root: &LcRoot<LigeroEncoding<F>, F>,
    proof: &LigeroEvalProof<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>,
    aux: LcProofAuxiliaryInfo,
    tr: &mut impl RemainderTranscript<F>,
    challenge_coord: &Vec<F>,
    claimed_value: F,
) where
    F: FieldExt,
{
    // --- Sanitycheck ---
    assert_eq!(aux.num_rows * aux.orig_num_cols, 1 << challenge_coord.len());

    // --- Grab the inner/outer tensors from the challenge point ---
    let (inner_tensor, outer_tensor) =
        get_ml_inner_outer_tensors(challenge_coord, aux.num_rows, aux.orig_num_cols);

    // --- Add the root to the transcript ---
    let _ = tr.append_field_element("root", root.root);

    // --- Reconstruct the encoding (TODO!(ryancao): Deprecate the encoding!) and verify ---
    let enc =
        LigeroEncoding::<F>::new_from_dims(proof.get_orig_num_cols(), proof.get_encoded_num_cols());
    let result = verify(
        &root.root,
        &outer_tensor[..],
        &inner_tensor[..],
        proof,
        &enc,
        tr,
    )
    .unwrap();

    assert_eq!(result, claimed_value);
}

#[cfg(test)]
pub mod tests {
    use crate::utils::get_random_coeffs_for_multilinear_poly;
    use crate::{
        adapter::convert_halo_to_lcpc,
        ligero_commit::{poseidon_ml_commit_prove, poseidon_ml_eval_prove},
        ligero_ml_helper::{get_ml_inner_outer_tensors, naive_eval_mle_at_challenge_point},
        verify,
    };
    use ark_std::test_rng;
    use itertools::Itertools;
    use rand::Rng;
    use remainder_shared_types::transcript::{
        poseidon_transcript::PoseidonTranscript, Transcript as RemainderTranscript,
    };
    use remainder_shared_types::Fr;
    use std::iter::repeat_with;

    use super::{
        remainder_ligero_commit_prove, remainder_ligero_eval_prove, remainder_ligero_verify,
    };

    /// This details what the Remainder interface should be doing.
    #[test]
    fn test_remainder_flow() {
        // --- Setup stuff ---
        let ml_num_vars = 16;
        let rho_inv = 4;
        let ratio = 1_f64;

        // --- Generate random polynomial ---
        let ml_coeffs = get_random_coeffs_for_multilinear_poly(ml_num_vars);
        let mut rng = test_rng();

        // --- Grab challenge point and claimed value ---
        let challenge_coord: Vec<Fr> = repeat_with(|| Fr::from(rng.gen::<u64>()))
            .take(ml_num_vars)
            .collect_vec();
        let claimed_value = naive_eval_mle_at_challenge_point(&ml_coeffs, &challenge_coord);
        let mut poseidon_transcript = PoseidonTranscript::new("Test transcript");

        // --- Commit, prove, convert ---
        let (_, comm, root, aux) = remainder_ligero_commit_prove(&ml_coeffs, rho_inv, ratio);
        let h2_ligero_proof: crate::adapter::LigeroProof<Fr> = remainder_ligero_eval_prove(
            &ml_coeffs,
            &challenge_coord,
            &mut poseidon_transcript,
            aux.clone(),
            comm,
            root.clone(),
        );
        let (_, ligero_eval_proof, _) = convert_halo_to_lcpc(aux.clone(), h2_ligero_proof);

        // --- Grab new Poseidon transcript + verify ---
        let mut verifier_poseidon_transcript = PoseidonTranscript::new("Test transcript");
        remainder_ligero_verify::<Fr>(
            &root,
            &ligero_eval_proof,
            aux,
            &mut verifier_poseidon_transcript,
            &challenge_coord,
            claimed_value,
        );
    }

    /// This basically details what a Ligero prover would be doing.
    #[test]
    fn test_commit_eval_proof() {
        // --- Hyperparams (to be determined) ---
        let ml_num_vars = 8;
        let log_num_rows = 4;
        let log_orig_num_cols = 4;
        let rho_inv = 4;
        let ratio = 1_f64;
        let ligero_root_filename = "ligero_root.txt";
        let ligero_proof_filename = "ligero_eval_proof.txt";
        let ligero_aux_data_filename = "ligero_aux_info.txt";
        let ligero_claim_filename = "ligero_claim_info.txt";

        let num_rows = 1 << log_num_rows;
        let orig_num_cols = 1 << log_orig_num_cols;

        // --- Generate random coeffs and random challenge point to evaluate at ---
        let mut rng = test_rng();
        let ml_coeffs = get_random_coeffs_for_multilinear_poly(ml_num_vars);
        let challenge_coord: Vec<Fr> = repeat_with(|| Fr::from(rng.gen::<u64>()))
            .take(ml_num_vars)
            .collect_vec();
        let correct_eval = naive_eval_mle_at_challenge_point(&ml_coeffs, &challenge_coord);

        // --- Commit phase ---
        let (enc, comm, root, _) = poseidon_ml_commit_prove::<Fr>(
            &ml_coeffs,
            log_num_rows,
            log_orig_num_cols,
            rho_inv,
            Some(ligero_root_filename),
        );

        // --- Eval phase ---
        // --- Initialize transcript (note that this would come from GKR) ---
        let mut poseidon_transcript = PoseidonTranscript::new("Test transcript");
        let (_eval, proof) = poseidon_ml_eval_prove(
            &ml_coeffs,
            rho_inv,
            log_num_rows,
            log_orig_num_cols,
            &challenge_coord,
            &mut poseidon_transcript,
            Some(ligero_proof_filename),
            Some(ligero_aux_data_filename),
            Some(ligero_claim_filename),
            comm,
            root.clone(),
        )
        .unwrap();

        // --- Verify phase ---
        let mut poseidon_transcript_verifier = PoseidonTranscript::new("Test transcript");
        let _ = poseidon_transcript_verifier.append_field_element("root", root.root);
        let (inner_tensor, outer_tensor) =
            get_ml_inner_outer_tensors(&challenge_coord, num_rows, orig_num_cols);
        let result = verify(
            &root.root,
            &outer_tensor,
            &inner_tensor,
            &proof,
            &enc,
            &mut poseidon_transcript_verifier,
        )
        .unwrap();

        assert_eq!(result, correct_eval);

        // let _ = load_and_convert::<Fr, Fr2>(
        //     &ligero_aux_data_filename,
        //     &ligero_proof_filename,
        //     &ligero_root_filename,
        // );
    }
}
