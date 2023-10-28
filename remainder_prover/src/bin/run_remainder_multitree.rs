//! Executable for Remainder ZKDT prover!

use ark_serialize::Read;
use clap::Parser;
use remainder_shared_types::Fr;

use remainder::{
    prover::{GKRCircuit, GKRError},
    zkdt::{
        constants::{
            get_sample_minibatch_commitment_filepath_for_batch_size, get_tree_commitment_filepath_for_tree_batch,
        },
        input_data_to_circuit_adapter::{
            convert_zkdt_circuit_data_into_mles, load_upshot_data_single_tree_batch, MinibatchData, BatchedZKDTCircuitMles,
        }, zkdt_multiple_tree_circuit::ZKDTMultiTreeCircuit,
    },
};

use remainder_shared_types::transcript::Transcript;
use remainder_shared_types::FieldExt;

use std::{
    fs,
    io::BufWriter,
    path::{Path, PathBuf},
    time::Instant,
};

use thiserror::Error;
use tracing::debug;
use tracing_subscriber::{
    fmt::format::{self, FmtSpan},
    prelude::*,
    FmtSubscriber,
};

use ark_std::{end_timer, start_timer};

#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;


#[derive(Error, Debug, Clone)]
/// Errors for running the binary over inputs and proving/verification
/// TODO!(ryancao): File-handling errors!
pub enum ZKDTBinaryError {
    #[error("GKR Verification failed! Error: {0}")]
    GKRVerificationFailed(GKRError),

    #[error("GKR Proving failed! Error: {0}")]
    GKRProvingFailed(GKRError),

    #[error("Passed in minibatch logsize without specifying minibatch number")]
    MinibatchLogsizeNoIndex,

    #[error("Input commitment file does not exist")]
    NoInputCommitmentFile,

    #[error("Tree commitment file does not exist")]
    NoTreeCommitmentFile,
}

/// Executable for running Remainder's GKR prover over the ZKDT circuit,
/// taking in as input the outputs from the Python processing script and
/// a pre-processed Ligero decision tree commitment.
///
/// Should be able to:
/// * Generate a proof over any tree and any set of data
/// * Verify that proof or not
/// * Write the proof or not to a filepath of choice
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the directory storing the tree commitment files
    #[arg(long)]
    tree_commit_dir: String,

    /// the batch size of trees we are combining
    #[arg(long, default_value_t = 4)]
    tree_batch_size: usize,

    /// The tree number we are actually running (note that this,
    /// alongside the `tree_commit_dir`, generates the filename
    /// for the actual tree commitment file we are using)
    #[arg(long)]
    tree_batch_number: usize,
    
    /// Path to the sample minibatch commitment directory containing
    /// all of the sample minibatch commitments.
    ///
    /// Note that the commitment filename will be autogenerated from
    /// this parameter, plus the given `log_sample_minibatch_size` and
    /// `sample_minibatch_number`.
    #[arg(long)]
    sample_minibatch_commit_dir: String,

    /// Filepath to the sample intermediates that we care about
    /// (i.e. the outputs of the Python processing)
    #[arg(long)]
    quantized_samples_filepath: String,

    /// Filepath to the actual tree model that we care about
    /// (i.e. the tree JSON outputs of the Python processing)
    ///
    /// TODO!(ryancao): Make this only need to take in the
    /// singular tree that we are supposed to be proving, not
    /// the contents of the entire forest.
    #[arg(long)]
    decision_forest_model_filepath: String,

    /// log_2 of the minibatch size. Note that passing in `None` here
    /// will result in the entire dataset being treated as a single
    /// minibatch.
    #[arg(long)]
    log_sample_minibatch_size: Option<usize>,

    /// The minibatch number we are generating a proof for.
    /// Note that if `log_sample_batch_size` is `Some` then
    /// this value cannot be `None`.
    #[arg(long)]
    sample_minibatch_number: Option<usize>,

    /// Filepath to where the final GKR proof should be written to.
    /// (Note that not passing in anything here will result in no proof
    /// being written at all.)
    #[arg(long)]
    gkr_proof_to_be_written_filepath: Option<String>,

    /// Whether we want the proof to be verified or not.
    #[arg(long, default_value_t = false)]
    verify_proof: bool,

    /// Whether we want debug tracing subscriber logs or not.
    /// By default, we use `DEBUG` as the subscriber level.
    ///
    /// TODO!(ryancao): Figure out `structopt` so we can pass in
    /// different trace levels
    #[arg(long, default_value_t = false)]
    debug_tracing_subscriber: bool,

    /// sets the value for rho_inv for the ligero commit
    #[arg(long, default_value_t = 4)]
    rho_inv: u8,

    /// sets the matrix ratio (orig_num_cols : num_rows) for ligero commit, will do the dimensions
    /// to achieve the ratio as close as possible
    #[arg(long, default_value_t = 1_f64)]
    matrix_ratio: f64,




    // --- NOTE: The below flags are all no-ops! ---
    // TODO!(ryancao, marsenis): Tie these to the actual optimization
    // flags after a refactor

    // /// Whether to turn on claim aggregation optimization which
    // /// reduces the number of V_i(l(x)) evaluations sent over to
    // /// the verifier, rather than the upper bound of
    // /// `num_claims * num_challenge_points + 1`
    // #[arg(long, default_value_t = true)]
    // claim_agg_reduced_number_vi_l_x_evaluations_optimization: bool,

    // /// Whether to turn on claim aggregation optimization which
    // /// attempts to a) group claims by source input layers, b) aggregate
    // /// those claims first into a resulting claim, and c) aggregate
    // /// all the resulting claims.
    // /// (This includes claim de-duplicating)
    // #[arg(long, default_value_t = true)]
    // claim_agg_group_claims_by_input_layer_optimization: bool,

    // /// Whether to turn on flattened layer optimization for V_i(l(x)).
    // #[arg(long, default_value_t = true)]
    // compute_vi_l_x_flattened_optimization: bool,

    // /// Whether to turn on common-variable V_i(l(x)) computation optimization.
    // #[arg(long, default_value_t = true)]
    // compute_vi_l_x_common_var_optimization: bool,
}

/// Runs the actual circuit on the witness data

pub fn run_zkdt_circuit<F: FieldExt, C: GKRCircuit<F>>(
    mut circuit: C,
    maybe_filepath_to_proof: Option<PathBuf>,
    verify_proof: bool,
) -> Result<(), ZKDTBinaryError>
where
    <C as GKRCircuit<F>>::Transcript: Sync,
{
    let mut transcript = C::Transcript::new("GKR Prover Transcript");
    let now = Instant::now();

    match circuit.prove(&mut transcript) {
        Ok(proof) => {
            println!(
                "proof generated successfully in {}!",
                now.elapsed().as_secs_f32()
            );

            // --- Write proof to file, if filepath exists ---
            if let Some(filepath_to_proof) = maybe_filepath_to_proof.clone() {
                println!(
                    "Writing the serialized ZKDT GKR proof to {:?}",
                    filepath_to_proof
                );

                let timer = start_timer!(|| "proof writer");

                let file = fs::File::create(filepath_to_proof).unwrap();
                let bw = BufWriter::new(file);
                serde_json::to_writer(bw, &proof).unwrap();

                end_timer!(timer);
            }

            let mut transcript = C::Transcript::new("GKR Verifier Transcript");
            let now = Instant::now();

            // --- Verify proof if asked for ---
            if verify_proof {
                // --- Grab proof from filepath, if exists; else grab from memory ---
                let proof = if let Some(filepath_to_proof) = maybe_filepath_to_proof {
                    let mut file = std::fs::File::open(&filepath_to_proof).unwrap();
                    let initial_buffer_size =
                        file.metadata().map(|m| m.len() as usize + 1).unwrap_or(0);
                    let mut bufreader = Vec::with_capacity(initial_buffer_size);
                    file.read_to_end(&mut bufreader).unwrap();
                    serde_json::de::from_slice(&bufreader[..]).unwrap()

                } else {
                    proof
                };
                match circuit.verify(&mut transcript, proof) {
                    Ok(_) => {
                        println!(
                            "Verification succeeded: takes {}!",
                            now.elapsed().as_secs_f32()
                        );
                        Ok(())
                    }
                    Err(err) => {
                        println!("Verify failed! Error: {err}");
                        Err(ZKDTBinaryError::GKRVerificationFailed(err))
                    }
                }
            } else {
                Ok(())
            }
        }
        Err(err) => {
            println!("Proof failed! Error: {err}");
            Err(ZKDTBinaryError::GKRProvingFailed(err))
        }
    }
}

/// Idea here is as follows:
/// * Take in as input the files which Ben's Rust code takes in
/// * Output as files the actual proof which gets generated from that code
fn main() -> Result<(), ZKDTBinaryError> {
    let args = Args::parse();

    // --- Tracing subscriber (i.e. outputs trace messages in stdout) if asked
    // for ---
    let formatter =
    // Construct a custom formatter for `Debug` fields
    format::debug_fn(|writer, field, value| write!(writer, "{}: {:#?}", field, value))
        // Use the `tracing_subscriber::MakeFmtExt` trait to wrap the
        // formatter so that a delimiter is added between fields.
        .delimited("\n");

    if args.debug_tracing_subscriber {
        let subscriber = FmtSubscriber::builder()
            .with_line_number(true)
            .with_max_level(tracing::Level::TRACE)
            .with_level(true)
            .with_span_events(FmtSpan::ENTER | FmtSpan::CLOSE)
            .with_ansi(false)
            .fmt_fields(formatter)
            // .pretty()
            .finish();
        let _default_guard = tracing::subscriber::set_global_default(subscriber);
    }

    // --- Log the args ---
    let args_as_string = format!("{:?}", args);
    debug!(args_as_string);

    // --- Sanitycheck (need minibatch number if we have batch size) + grabbing minibatch data ---
    
    let maybe_minibatch_data = match (args.log_sample_minibatch_size, args.sample_minibatch_number)
    {
        (None, None) => None,
        (None, Some(_)) => None,
        (Some(_), None) => {
            return Err(ZKDTBinaryError::MinibatchLogsizeNoIndex);
        }

        (Some(log_sample_minibatch_size), Some(sample_minibatch_number)) => {
            let minibatch_data = MinibatchData {
                log_sample_minibatch_size,
                sample_minibatch_number,
            };
            Some(minibatch_data)
        }
    };


    let tree_range = (args.tree_batch_number * args.tree_batch_size) .. ((args.tree_batch_number + 1) * args.tree_batch_size);

    let (trees_batched_data, minibatch_data_vec): (Vec<BatchedZKDTCircuitMles<Fr>>, Vec<_>) = tree_range.into_iter().map(
        |tree_num| {
             // --- Read in the Upshot data from file ---
            let (zkdt_circuit_data, (tree_height, input_len), minibatch_data) =
            load_upshot_data_single_tree_batch::<Fr>(
                maybe_minibatch_data.clone(),
                tree_num,
                Path::new(&args.decision_forest_model_filepath),
                Path::new(&args.quantized_samples_filepath),
            );
            let (batched_catboost_mles, (_, _)) =
                convert_zkdt_circuit_data_into_mles(zkdt_circuit_data, tree_height, input_len);
            (batched_catboost_mles, minibatch_data)
        }
    ).unzip();


    // // --- Read in the Upshot data from file ---
    // let (zkdt_circuit_data, (tree_height, input_len), minibatch_data) =
    //     load_upshot_data_single_tree_batch::<Fr>(
    //         maybe_minibatch_data,
    //         args.tree_number,
    //         Path::new(&args.decision_forest_model_filepath),
    //         Path::new(&args.quantized_samples_filepath),
    //     );
    // let (batched_catboost_mles, (_, _)) =
    //     convert_zkdt_circuit_data_into_mles(zkdt_circuit_data, tree_height, input_len);

    // --- Sanitycheck (grab the minibatch commitment filename + check if exists) ---

let sample_minibatch_commitment_filepath =
        get_sample_minibatch_commitment_filepath_for_batch_size(
            minibatch_data_vec[0].log_sample_minibatch_size,
            minibatch_data_vec[0].sample_minibatch_number,
            Path::new(&args.sample_minibatch_commit_dir),
        );
    debug!(
        sample_minibatch_commitment_filepath,
        "Attempting to find the sample minibatch commitment file"
    );
    if let Err(_) = fs::metadata(&sample_minibatch_commitment_filepath) {
        return Err(ZKDTBinaryError::NoInputCommitmentFile);
    }

    // --- Sanitycheck (check if the tree commitment exists) ---
    
    let tree_commit_filepath = get_tree_commitment_filepath_for_tree_batch(
        args.tree_batch_size,
        args.tree_batch_number,
        Path::new(&args.tree_commit_dir),
    );
    debug!(
        tree_commit_filepath,
        "Attempting to find the tree commit file"
    );
    if let Err(_) = fs::metadata(&tree_commit_filepath) {
        return Err(ZKDTBinaryError::NoTreeCommitmentFile);
    }

    // --- Create the full ZKDT circuit ---
    // multi tree ZKDT circuit
    let full_zkdt_circuit = ZKDTMultiTreeCircuit {
        batched_zkdt_circuit_mles_tree: trees_batched_data,
        tree_precommit_filepath: tree_commit_filepath,
        sample_minibatch_precommit_filepath: sample_minibatch_commitment_filepath,
        rho_inv: args.rho_inv,
        ratio: args.matrix_ratio
    };

    // --- Grab the proof filepath to write to and compute the circuit + prove ---
    let maybe_proof_filepath = args
        .gkr_proof_to_be_written_filepath
        .map(|maybe_path| Path::new(&maybe_path).to_owned());
    run_zkdt_circuit(full_zkdt_circuit, maybe_proof_filepath, args.verify_proof)
}
