#[cfg(test)]
mod tests {
    use std::{path::Path};

    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use ark_std::{test_rng};
    
    use rand::Rng;

    use crate::{zkdt::{data_pipeline::dummy_data_generator::{BatchedCatboostMles, generate_mles_batch_catboost_single_tree}, multiset_circuit::legacy_circuits::MultiSetCircuit}};
    use remainder_shared_types::transcript::{Transcript};
    use crate::prover::tests::test_circuit;

    use super::super::circuits::{FSMultiSetCircuit};

    #[test]
    fn test_multiset_circuit_catboost_batched() {

        let mut rng = test_rng();

        let (BatchedCatboostMles {decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            multiplicities_bin_decomp_mle_decision,
            multiplicities_bin_decomp_mle_leaf,
            decision_nodes_mle,
            leaf_nodes_mle, ..}, (_tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

        let circuit = MultiSetCircuit::new(
            decision_nodes_mle,
            leaf_nodes_mle,
            multiplicities_bin_decomp_mle_decision,
            multiplicities_bin_decomp_mle_leaf,
            decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            Fr::from(rng.gen::<u64>()),
            (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())),
        );

        test_circuit(circuit, None);
    }

    // --- Note: This test will only compile/run if we have the FS input layer stuff ---
    // TODO!(ryancao): Unfortunately I do not have expertise in this area to implement at the moment :(
    // #[test]
    // fn test_fs_multiset_circuit_catboost_batched() {

    //     let (BatchedCatboostMles {decision_node_paths_mle_vec,
    //         leaf_node_paths_mle_vec,
    //         multiplicities_bin_decomp_mle_decision,
    //         multiplicities_bin_decomp_mle_leaf,
    //         decision_nodes_mle,
    //         leaf_nodes_mle, ..}, (_tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

    //     let circuit = FSMultiSetCircuit::new(
    //         decision_nodes_mle,
    //         leaf_nodes_mle,
    //         multiplicities_bin_decomp_mle_decision,
    //         multiplicities_bin_decomp_mle_leaf,
    //         decision_node_paths_mle_vec,
    //         leaf_node_paths_mle_vec,
    //     );

    //     test_circuit(circuit, None);
    // }

}