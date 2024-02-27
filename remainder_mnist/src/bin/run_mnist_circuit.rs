//! Executable for Remainder ZKDT prover!

use ark_serialize::Read;
use clap::Parser;
use remainder_shared_types::Fr;

use remainder::{
    prover::{GKRCircuit, GKRError},
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

#[derive(Error, Debug, Clone)]
/// Errors for running the binary over inputs and proving/verification
pub enum MatMulBinaryError {
    #[error("GKR Verification failed! Error: {0}")]
    GKRVerificationFailed(GKRError),

    #[error("GKR Proving failed! Error: {0}")]
    GKRProvingFailed(GKRError),

    #[error("Input commitment file does not exist")]
    NoInputCommitmentFile,
}

/// Executable for running Remainder's GKR prover over the MNIST circuit,
/// taking in as input the quantized weights/biases the the MNIST model
///
/// Should be able to:
/// * Generate a proof over the circuit
/// * Produce a prediction given the MNIST model output 
/// * Verify that proof or not
/// * Write the proof or not to a filepath of choice
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {

    /// Filepath to the sample intermediates that we care about
    /// (i.e. the outputs of the Python processing)
    #[arg(long)]
    quantized_samples_filepath: String,

    /// Filepath to the actual MNIST model that we care about
    /// (i.e. the MNIST JSON outputs of the Python processing)
    #[arg(long)]
    mnist_model_filepath: String,

    /// Filepath to where the final GKR proof should be written to.
    /// (Note that not passing in anything here will result in no proof
    /// being written at all.)
    #[arg(long)]
    gkr_proof_to_be_written_filepath: Option<String>,

    /// Mock proving with dummy data and weights or not.
    #[arg(long, default_value_t = true)]
    use_dummy_data_and_weights: bool,

    /// Whether we want the proof to be verified or not.
    #[arg(long, default_value_t = false)]
    verify_proof: bool,

    /// Whether we want debug tracing subscriber logs or not.
    /// By default, we use `DEBUG` as the subscriber level.
    #[arg(long, default_value_t = false)]
    debug_tracing_subscriber: bool,

    /// Whether we want info tracing subscriber logs or not.
    /// By default, we use `INFO` as the subscriber level.
    #[arg(long, default_value_t = false)]
    info_tracing_subscriber: bool,

    /// sets the value for rho_inv for the ligero commit
    #[arg(long, default_value_t = 4)]
    rho_inv: u8,

    /// sets the matrix ratio (orig_num_cols : num_rows) for ligero commit, will do the dimensions
    /// to achieve the ratio as close as possible
    #[arg(long, default_value_t = 1_f64)]
    matrix_ratio: f64,
}

/// Runs the actual circuit on the witness data

pub fn run_mnist_circuit<F: FieldExt, C: GKRCircuit<F>>(
    mut circuit: C,
    maybe_filepath_to_proof: Option<PathBuf>,
    verify_proof: bool,
    tree_batch_size: usize,
) -> Result<(), MatMulBinaryError>
where
    <C as GKRCircuit<F>>::Transcript: Sync,
{
    todo!()
    // let mut transcript = C::Transcript::new("GKR Prover Transcript");
    // let now = Instant::now();

    // match circuit.prove(&mut transcript) {
    //     Ok(proof) => {
    //         println!(
    //             "proof generated successfully in {}!",
    //             now.elapsed().as_secs_f32()
    //         );

    //         let zkdt_proof = ZKDTProof {
    //             gkr_proof: proof,
    //             tree_batch_size,
    //         };

    //         // --- Write proof to file, if filepath exists ---
    //         if let Some(filepath_to_proof) = maybe_filepath_to_proof.clone() {
    //             println!(
    //                 "Writing the serialized ZKDT GKR proof to {:?}",
    //                 filepath_to_proof
    //             );

    //             let timer = start_timer!(|| "proof writer");

    //             let file = fs::File::create(filepath_to_proof).unwrap();
    //             let bw = BufWriter::new(file);
    //             serde_json::to_writer(bw, &zkdt_proof).unwrap();

    //             end_timer!(timer);
    //         }

    //         let mut transcript = C::Transcript::new("GKR Verifier Transcript");
    //         let now = Instant::now();

    //         // --- Verify proof if asked for ---
    //         if verify_proof {
    //             // --- Grab proof from filepath, if exists; else grab from memory ---
    //             let proof = if let Some(filepath_to_proof) = maybe_filepath_to_proof {
    //                 let mut file = std::fs::File::open(&filepath_to_proof).unwrap();
    //                 let initial_buffer_size =
    //                     file.metadata().map(|m| m.len() as usize + 1).unwrap_or(0);
    //                 let mut bufreader = Vec::with_capacity(initial_buffer_size);
    //                 file.read_to_end(&mut bufreader).unwrap();
    //                 let zkdt_proof: ZKDTProof<F, C::Transcript> =
    //                     serde_json::de::from_slice(&bufreader[..]).unwrap();
    //                 zkdt_proof.gkr_proof
    //             } else {
    //                 zkdt_proof.gkr_proof
    //             };
    //             match circuit.verify(&mut transcript, proof) {
    //                 Ok(_) => {
    //                     println!(
    //                         "Verification succeeded: takes {}!",
    //                         now.elapsed().as_secs_f32()
    //                     );
    //                     Ok(())
    //                 }
    //                 Err(err) => {
    //                     println!("Verify failed! Error: {err}");
    //                     Err(MatMulBinaryError::GKRVerificationFailed(err))
    //                 }
    //             }
    //         } else {
    //             Ok(())
    //         }
    //     }
    //     Err(err) => {
    //         println!("Proof failed! Error: {err}");
    //         Err(MatMulBinaryError::GKRProvingFailed(err))
    //     }
    // }
}

/// Idea here is as follows:
/// * Take in as input the files which Ben's Rust code takes in
/// * Output as files the actual proof which gets generated from that code
fn main() -> Result<(), MatMulBinaryError> {
    let args = Args::parse();

    // --- Tracing subscriber (i.e. outputs trace messages in stdout) if asked
    // for ---
    let formatter =
    // Construct a custom formatter for `Debug` fields
    format::debug_fn(|writer, field, value| write!(writer, "{}: {:#?}", field, value))
        // Use the `tracing_subscriber::MakeFmtExt` trait to wrap the
        // formatter so that a delimiter is added between fields.
        .delimited("\n");

    if args.debug_tracing_subscriber || args.info_tracing_subscriber {
        let subscriber = FmtSubscriber::builder()
            .with_line_number(true)
            .with_max_level(if args.debug_tracing_subscriber {tracing::Level::DEBUG} else {tracing::Level::INFO})
            .with_level(true)
            .with_span_events(FmtSpan::ENTER | FmtSpan::CLOSE)
            .with_ansi(false)
            .fmt_fields(formatter)
            // .pretty()
            .finish();
        let _default_guard = tracing::subscriber::set_global_default(subscriber);
    }

    // // --- Log the args ---
    let args_as_string = format!("{:?}", args);
    debug!(args_as_string);

    let (trees_batched_data, (tree_height, input_len), minibatch_data) =
        load_upshot_data_multi_tree_batch::<Fr>(
            maybe_minibatch_data.clone(),
            args.tree_batch_size,
            args.tree_batch_number,
            Path::new(&args.decision_forest_model_filepath),
            Path::new(&args.quantized_samples_filepath),
        );

    // // --- Sanitycheck (check if the tree commitment exists) ---

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
        return Err(MatMulBinaryError::NoTreeCommitmentFile);
    }

    // --- Create the full ZKDT circuit ---
    // multi tree ZKDT circuit
    let full_zkdt_circuit = ZKDTMultiTreeCircuit {
        batched_zkdt_circuit_mles_tree: tree_batched_circuit_mles,
        tree_precommit_filepath: tree_commit_filepath,
        rho_inv: args.rho_inv,
        ratio: args.matrix_ratio,
    };

    // // --- Grab the proof filepath to write to and compute the circuit + prove ---
    // let maybe_proof_filepath = args
    //     .gkr_proof_to_be_written_filepath
    //     .map(|maybe_path| Path::new(&maybe_path).to_owned());
    // run_mnist_circuit(
    //     full_zkdt_circuit,
    //     maybe_proof_filepath,
    //     args.verify_proof,
    //     args.tree_batch_size,
    // )
    todo!()
}
