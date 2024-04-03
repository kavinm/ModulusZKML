//! Executable for demo MLP Remainder circuit solution!

use ark_serialize::Read;
use clap::Parser;
use remainder_mlp::soln_workshop_exercise_circuit::soln_nn_full_circuit::MLPCircuit;
use remainder_mlp::soln_workshop_exercise_circuit::soln_workshop_utils::load_dummy_mlp_input_and_weights;

use remainder::prover::{GKRCircuit, GKRError, GKRProof};

use remainder_shared_types::FieldExt;
use remainder_shared_types::{transcript::Transcript, Fr};

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
pub enum MLPBinaryError {
    #[error("GKR Verification failed! Error: {0}")]
    GKRVerificationFailed(GKRError),

    #[error("GKR Proving failed! Error: {0}")]
    GKRProvingFailed(GKRError),
}

/// Executable for running Remainder's GKR prover over an MLP
/// FC + ReLU circuit
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input dim length
    #[arg(long)]
    input_dim: usize,

    /// Output dim length
    #[arg(long)]
    output_dim: usize,

    /// List of hidden dims
    #[arg(long)]
    hidden_dim: usize,

    /// Whether we want the proof to be verified or not.
    #[arg(long, default_value_t = false)]
    verify_proof: bool,

    /// Filepath to where the final GKR proof should be written to.
    /// (Note that not passing in anything here will result in no proof
    /// being written at all.)
    #[arg(long)]
    gkr_proof_to_be_written_filepath: Option<String>,

    /// Whether we want debug tracing subscriber logs or not.
    /// By default, we use `DEBUG` as the subscriber level.
    ///
    /// TODO!(ryancao): Figure out `structopt` so we can pass in
    /// different trace levels
    #[arg(long, default_value_t = false)]
    debug_tracing_subscriber: bool,

    /// Whether we want info tracing subscriber logs or not.
    /// By default, we use `INFO` as the subscriber level.
    ///
    /// Note that if `debug_tracing_subscriber` is also `true`,
    /// we will set the tracing subscriber to `DEBUG` (always
    /// use the more detailed of the two).
    #[arg(long, default_value_t = false)]
    info_tracing_subscriber: bool,
}

/// Runs the actual circuit on the witness data
pub fn run_mlp_circuit<F: FieldExt, C: GKRCircuit<F>>(
    mut circuit: C,
    maybe_filepath_to_proof: Option<PathBuf>,
    verify_proof: bool,
) -> Result<(), MLPBinaryError>
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
                    let proof: GKRProof<F, C::Transcript> =
                        serde_json::de::from_slice(&bufreader[..]).unwrap();
                    proof
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
                        Err(MLPBinaryError::GKRVerificationFailed(err))
                    }
                }
            } else {
                Ok(())
            }
        }
        Err(err) => {
            println!("Proof failed! Error: {err}");
            Err(MLPBinaryError::GKRProvingFailed(err))
        }
    }
}

fn main() -> Result<(), MLPBinaryError> {
    let args = Args::parse();

    // --- Tracing subscriber (i.e. outputs trace messages in stdout) if asked for ---
    let formatter =
    // Construct a custom formatter for `Debug` fields
    format::debug_fn(|writer, field, value| write!(writer, "{}: {:#?}", field, value))
        // Use the `tracing_subscriber::MakeFmtExt` trait to wrap the
        // formatter so that a delimiter is added between fields.
        .delimited("\n");

    if args.debug_tracing_subscriber || args.info_tracing_subscriber {
        let subscriber = FmtSubscriber::builder()
            .with_line_number(true)
            .with_max_level(if args.debug_tracing_subscriber {
                tracing::Level::DEBUG
            } else {
                tracing::Level::INFO
            })
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

    // --- Create the actual linear model ---
    let (mnist_weights, mnist_inputs) =
        load_dummy_mlp_input_and_weights::<Fr>(args.input_dim, args.hidden_dim, args.output_dim);

    let circuit = MLPCircuit::<Fr>::new(mnist_weights, mnist_inputs);

    // --- Grab the proof filepath to write to and compute the circuit + prove ---
    let maybe_proof_filepath = args
        .gkr_proof_to_be_written_filepath
        .map(|maybe_path| Path::new(&maybe_path).to_owned());
    run_mlp_circuit(circuit, maybe_proof_filepath, args.verify_proof)
}
