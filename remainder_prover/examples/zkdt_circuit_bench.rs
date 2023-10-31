use std::{fs, path::Path, time::Instant};

use remainder_shared_types::Fr;
use remainder::{
    prover::GKRCircuit,
    zkdt::{
        zkdt_circuit::ZKDTCircuit, cache_upshot_catboost_inputs_for_testing::generate_mles_batch_catboost_single_tree,
    },
};
use remainder_shared_types::{transcript::Transcript, FieldExt};
use serde_json::{from_reader, to_writer};
use remainder::prover::helpers::test_circuit;

fn main() {
    // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
    // tracing::subscriber::set_global_default(subscriber)
    //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

    let batch_size = 9;

    let (batched_catboost_mles, (_, _)) =
        generate_mles_batch_catboost_single_tree::<Fr>(batch_size, Path::new("upshot_data"));

    let combined_circuit = ZKDTCircuit {
        batched_zkdt_circuit_mles: batched_catboost_mles,
        tree_precommit_filepath: "upshot_data/tree_ligero_commitments/tree_commitment_0.json"
            .to_string(),
        sample_minibatch_precommit_filepath: "upshot_data/sample_minibatch_commitments/sample_minibatch_logsize_1_commitment_0.json".to_string(),
    };

    //Use this code to get the circuit hash for your circuit
    // dbg!(combined_circuit.gen_circuit_hash().to_bytes());

    test_circuit(combined_circuit, Some(Path::new("./zkdt_proof.json")));
}
