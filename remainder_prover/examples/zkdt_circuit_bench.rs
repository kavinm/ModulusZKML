use std::{path::Path, time::Instant, fs};

use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use remainder::{zkdt::{zkdt_circuit::CombinedCircuits, data_pipeline::{dt2zkdt::load_upshot_data_single_tree_batch, dummy_data_generator::generate_mles_batch_catboost_single_tree}}, prover::GKRCircuit};
use remainder_shared_types::{FieldExt, transcript::Transcript};
use serde_json::{to_writer, from_reader};
use tracing::Level;

pub fn test_circuit<F: FieldExt, C: GKRCircuit<F>>(mut circuit: C, path: Option<&Path>) {
    let mut transcript = C::Transcript::new("GKR Prover Transcript");
    let now = Instant::now();

    match circuit.prove(&mut transcript) {
        Ok(proof) => {
            println!(
                "proof generated successfully in {}!",
                now.elapsed().as_secs_f32()
            );
            if let Some(path) = path {
                let mut f = fs::File::create(path).unwrap();
                to_writer(&mut f, &proof).unwrap();
            }
            let mut transcript = C::Transcript::new("GKR Verifier Transcript");
            let now = Instant::now();

            let proof = if let Some(path) = path {
                let file = std::fs::File::open(path).unwrap();

                from_reader(&file).unwrap()
            } else {
                proof
            };
            match circuit.verify(&mut transcript, proof) {
                Ok(_) => {
                    println!(
                        "Verification succeeded: takes {}!",
                        now.elapsed().as_secs_f32()
                    );
                }
                Err(err) => {
                    println!("Verify failed! Error: {err}");
                    panic!();
                }
            }
        }
        Err(err) => {
            println!("Proof failed! Error: {err}");
            panic!();
        }
    }
}

fn main() {
    // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
    // tracing::subscriber::set_global_default(subscriber)
    //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

    let batch_size = 10;

    let (batched_catboost_mles, (_, _)) = generate_mles_batch_catboost_single_tree::<Fr>(batch_size, todo!());

    // let combined_circuit = CombinedCircuits {
    //     batched_catboost_mles
    // };

    // test_circuit(combined_circuit, Some(Path::new("./zkdt_proof.json")));
}