#[cfg(test)]
mod tests {
    use std::{time::Instant, path::Path};
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use crate::{zkdt::{data_pipeline::dummy_data_generator::{BatchedCatboostMles, generate_mles_batch_catboost_single_tree}, bits_are_binary_circuit::{circuits::{BinDecomp16BitIsBinaryCircuit, BinDecomp4BitIsBinaryCircuit}, dataparallel_circuits::{BinDecomp16BitIsBinaryCircuitBatched, BinDecomp4BitIsBinaryCircuitBatched}}}, prover::GKRCircuit};
    use remainder_shared_types::transcript::{Transcript, poseidon_transcript::PoseidonTranscript};

    #[test]
    fn test_bin_decomp_16_bit_bits_are_binary() {

        // --- NOTE that this won't work unless we flip the binary decomp endian-ness!!! ---
        // let DummyMles { 
        //     dummy_permuted_input_data_mle,
        //     dummy_decision_node_paths_mle,
        //     dummy_binary_decomp_diffs_mle,
        //     ..
        // } = generate_dummy_mles();

        let (BatchedCatboostMles {
            binary_decomp_diffs_mle_vec, 
            ..
        }, (_tree_height, _)) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data"));

        let mut circuit = BinDecomp16BitIsBinaryCircuit::<Fr>::new(
            binary_decomp_diffs_mle_vec[0].clone(),
        );

        let mut transcript = PoseidonTranscript::new("Bin Decomp Bits Are Binary Transcript");
        let now = Instant::now();
        let proof = circuit.prove(&mut transcript);
        println!("Proof generated!: Took {} seconds", now.elapsed().as_secs_f32());

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Bin Decomp Bits Are Binary Transcript");
                let result = circuit.verify(&mut transcript, proof);
                if let Err(err) = result {
                    println!("{}", err);
                    panic!();
                }
            },
            Err(err) => {
                println!("{}", err);
                panic!();
            }
        }
    }

    #[test]
    fn c() {

        // --- NOTE that this won't work unless we flip the binary decomp endian-ness!!! ---
        // let DummyMles { 
        //     dummy_permuted_input_data_mle,
        //     dummy_decision_node_paths_mle,
        //     dummy_binary_decomp_diffs_mle,
        //     ..
        // } = generate_dummy_mles();

        let (BatchedCatboostMles {
            binary_decomp_diffs_mle_vec,
            ..
        }, (_tree_height, _)) = generate_mles_batch_catboost_single_tree::<Fr>(6, Path::new("upshot_data"));

        let mut circuit = BinDecomp16BitIsBinaryCircuitBatched::new(
            binary_decomp_diffs_mle_vec,
        );

        let mut transcript = PoseidonTranscript::new("Bin Decomp Bits Are Binary Transcript Batched");
        let now = Instant::now();
        let proof = circuit.prove(&mut transcript);
        println!("Proof generated!: Took {} seconds", now.elapsed().as_secs_f32());

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Bin Decomp Bits Are Binary Transcript Batched");
                let result = circuit.verify(&mut transcript, proof);
                if let Err(err) = result {
                    println!("{}", err);
                    panic!();
                }
            },
            Err(err) => {
                println!("{}", err);
                panic!();
            }
        }
    }

    #[test]
    fn test_bin_decomp_4_bit_bits_are_binary() {

        // --- NOTE that this won't work unless we flip the binary decomp endian-ness!!! ---
        // let DummyMles { 
        //     dummy_permuted_input_data_mle,
        //     dummy_decision_node_paths_mle,
        //     dummy_binary_decomp_diffs_mle,
        //     ..
        // } = generate_dummy_mles();

        let (BatchedCatboostMles {
            multiplicities_bin_decomp_mle_input_vec, 
            ..
        }, (_tree_height, _)) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data"));

        let mut circuit = BinDecomp4BitIsBinaryCircuit::<Fr>::new(
            multiplicities_bin_decomp_mle_input_vec[0].clone(),
        );

        let mut transcript = PoseidonTranscript::new("Bin Decomp Bits Are Binary Transcript");
        let now = Instant::now();
        let proof = circuit.prove(&mut transcript);
        println!("Proof generated!: Took {} seconds", now.elapsed().as_secs_f32());

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Bin Decomp Bits Are Binary Transcript");
                let result = circuit.verify(&mut transcript, proof);
                if let Err(err) = result {
                    println!("{}", err);
                    panic!();
                }
            },
            Err(err) => {
                println!("{}", err);
                panic!();
            }
        }
    }

    #[test]
    fn test_bin_decomp_4_bit_bits_are_binary_batched() {

        // --- NOTE that this won't work unless we flip the binary decomp endian-ness!!! ---
        // let DummyMles { 
        //     dummy_permuted_input_data_mle,
        //     dummy_decision_node_paths_mle,
        //     dummy_binary_decomp_diffs_mle,
        //     ..
        // } = generate_dummy_mles();

        let (BatchedCatboostMles {
            multiplicities_bin_decomp_mle_input_vec,
            ..
        }, (_tree_height, _)) = generate_mles_batch_catboost_single_tree::<Fr>(6, Path::new("upshot_data"));

        let mut circuit = BinDecomp4BitIsBinaryCircuitBatched::new(
            multiplicities_bin_decomp_mle_input_vec,
        );

        let mut transcript = PoseidonTranscript::new("Bin Decomp Bits Are Binary Transcript Batched");
        let now = Instant::now();
        let proof = circuit.prove(&mut transcript);
        println!("Proof generated!: Took {} seconds", now.elapsed().as_secs_f32());

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Bin Decomp Bits Are Binary Transcript Batched");
                let result = circuit.verify(&mut transcript, proof);
                if let Err(err) = result {
                    println!("{}", err);
                    panic!();
                }
            },
            Err(err) => {
                println!("{}", err);
                panic!();
            }
        }
    }

}