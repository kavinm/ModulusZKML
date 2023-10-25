use ark_serialize::Read;
use itertools::{repeat_n, Itertools};
use remainder_ligero::{
    ligero_structs::LigeroEncoding, poseidon_ligero::PoseidonSpongeHasher, LcCommit,
    LcProofAuxiliaryInfo, LcRoot,
};
use serde_json::from_reader;

use crate::prover::{GKRCircuit, Layers, Witness};
use crate::{
    layer::{
        batched::{combine_zero_mle_ref, unbatch_mles, BatchedLayer},
        empty_layer::EmptyLayer,
        LayerBuilder, LayerId, Padding,
    },
    mle::{dense::DenseMle, Mle, MleIndex, MleRef},
    prover::{
        combine_layers::combine_layers,
        input_layer::{
            self,
            combine_input_layers::InputLayerBuilder,
            enum_input_layer::{CommitmentEnum, InputLayerEnum},
            ligero_input_layer::LigeroInputLayer,
            public_input_layer::PublicInputLayer,
            random_input_layer::RandomInputLayer,
            InputLayer, MleInputLayer,
        },
        GKRError,
    },
    zkdt::builders::{AttributeConsistencyBuilderZeroRef, BitExponentiationBuilderCatBoost},
};
use remainder_shared_types::{
    transcript::{poseidon_transcript::PoseidonTranscript, Transcript},
    FieldExt,
};

use ark_std::{end_timer, start_timer};

use super::attribute_consistency_circuit::multitree_circuits::AttributeConsistencyCircuitMultiTree;
use super::binary_recomp_circuit::multitree_circuits::BinaryRecompCircuitMultiTree;
use super::bits_are_binary_circuit::multitree_circuits::{BinDecomp4BitIsBinaryCircuitBatchedMultiTree, BinDecomp16BitIsBinaryCircuitBatchedMultiTree, BinDecomp16BitIsBinaryCircuitMultiTree};
use super::input_data_to_circuit_adapter::BatchedZKDTCircuitMles;
use super::input_multiset_circuit::multitree_circuits::InputMultiSetCircuitMultiTree;
use super::multiset_circuit::multitree_circuits::FSMultiSetCircuitMultiTree;
use super::{
    attribute_consistency_circuit::dataparallel_circuits::AttributeConsistencyCircuit,
    binary_recomp_circuit::dataparallel_circuits::BinaryRecompCircuitBatched,
    bits_are_binary_circuit::{
        circuits::BinDecomp16BitIsBinaryCircuit,
        dataparallel_circuits::{
            BinDecomp16BitIsBinaryCircuitBatched, BinDecomp4BitIsBinaryCircuitBatched,
        },
    },
    input_multiset_circuit::dataparallel_circuits::InputMultiSetCircuit,
    multiset_circuit::circuits::FSMultiSetCircuit,
    path_consistency_circuit::circuits::PathCheckCircuitBatchedMul,
    structs::{BinDecomp16Bit, BinDecomp4Bit, DecisionNode, InputAttribute, LeafNode},
};

use std::io::BufReader;
use std::{marker::PhantomData, path::Path};

/// The actual ZKDT circuit!
pub struct ZKDTMultiTreeCircuit<F: FieldExt> {
    /// All of the input MLEs coming from the data generation pipeline
    pub batched_zkdt_circuit_mles_tree: Vec<BatchedZKDTCircuitMles<F>>,
    /// The filepath to the precommitted tree that we are proving
    pub tree_precommit_filepath: String,
    /// The filepath to the precommitted sample minibatch that we are proving
    pub sample_minibatch_precommit_filepath: String,
    /// rho inverse value for ligero commit
    pub rho_inv: u8,
    /// ratio
    pub ratio: f64,
}

impl<F: FieldExt> GKRCircuit<F> for ZKDTMultiTreeCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    // Uncomment this to turn on the circuit hash. Just make sure the hash you use is accurate to your batch size.
    // This one is for a batch size of 2^9
    // const CIRCUIT_HASH: Option<[u8; 32]> = Some([
    //     244, 174, 223, 136, 11, 9, 112, 40, 60, 180, 81, 61, 132, 165, 170, 36,
    //     31, 16, 66, 9, 54, 240, 75, 246, 68, 30, 31, 209, 242, 106, 147, 41,
    // ]);

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }

    fn synthesize_and_commit(
        &mut self,
        transcript: &mut Self::Transcript,
    ) -> Result<
        (
            Witness<F, Self::Transcript>,
            Vec<input_layer::enum_input_layer::CommitmentEnum<F>>,
        ),
        crate::prover::GKRError,
    > {
        let create_sub_circuits_timer = start_timer!(|| "input + instantiate sub circuits");
        let (
            // --- Actual circuit components ---
            mut attribute_consistency_circuit,
            mut multiset_circuit,
            mut input_multiset_circuit,
            mut binary_recomp_circuit_batched,
            mut path_consistency_circuit_batched,
            mut bin_decomp_4_bit_batched_bits_binary,
            mut bin_decomp_16_bit_batched_bits_binary,
            mut bits_are_binary_multiset_decision_circuit,
            mut bits_are_binary_multiset_leaf_circuit,
            // --- Input layer ---
            input_layers,
            inpunt_layers_commits,
        ) = self.create_sub_circuits(transcript).unwrap();
        end_timer!(create_sub_circuits_timer);

        let wit_gen_timer = start_timer!(|| "witness generation of subcircuits");
        let attribute_consistency_witness = attribute_consistency_circuit.yield_sub_circuit();
        let multiset_witness = multiset_circuit.yield_sub_circuit();
        let input_multiset_witness = input_multiset_circuit.yield_sub_circuit();
        let binary_recomp_circuit_batched_witness =
            binary_recomp_circuit_batched.yield_sub_circuit();
        let bin_decomp_4_bit_binary_batched_witness =
            bin_decomp_4_bit_batched_bits_binary.yield_sub_circuit();
        let bin_decomp_16_bit_binary_batched_witness =
            bin_decomp_16_bit_batched_bits_binary.yield_sub_circuit();
        let bits_are_binary_multiset_decision_circuit_witness =
            bits_are_binary_multiset_decision_circuit.yield_sub_circuit();
        let bits_are_binary_multiset_leaf_circuit_witness =
            bits_are_binary_multiset_leaf_circuit.yield_sub_circuit();
        end_timer!(wit_gen_timer);

        let combine_layers_timer = start_timer!(|| "combine layers + gate stuff");
        let (mut combined_circuit_layers, combined_circuit_output_layers) = combine_layers(
            vec![
                attribute_consistency_witness.layers,
                multiset_witness.layers,
                // input_multiset_witness.layers,
                // binary_recomp_circuit_batched_witness.layers,
                bin_decomp_4_bit_binary_batched_witness.layers,
                bin_decomp_16_bit_binary_batched_witness.layers,
                bits_are_binary_multiset_decision_circuit_witness.layers,
                bits_are_binary_multiset_leaf_circuit_witness.layers,
            ],
            vec![
                attribute_consistency_witness.output_layers,
                multiset_witness.output_layers,
                // input_multiset_witness.output_layers,
                // binary_recomp_circuit_batched_witness.output_layers,
                bin_decomp_4_bit_binary_batched_witness.output_layers,
                bin_decomp_16_bit_binary_batched_witness.output_layers,
                bits_are_binary_multiset_decision_circuit_witness.output_layers,
                bits_are_binary_multiset_leaf_circuit_witness.output_layers,
            ],
        )
        .unwrap();

        // --- Manually add the layers and output layers from the circuit involving gate MLEs ---
        // let updated_combined_output_layers = path_consistency_circuit_batched
        //     .add_subcircuit_layers_to_combined_layers(
        //         &mut combined_circuit_layers,
        //         combined_circuit_output_layers,
        //     );

        end_timer!(combine_layers_timer);

        Ok((
            Witness {
                layers: combined_circuit_layers,
                output_layers: combined_circuit_output_layers,
                // output_layers: updated_combined_output_layers,
                input_layers,
            },
            inpunt_layers_commits,
        ))
    }
}

impl<F: FieldExt> ZKDTMultiTreeCircuit<F> {
    fn create_sub_circuits(
        &mut self,
        transcript: &mut PoseidonTranscript<F>,
    ) -> Result<
        (
            AttributeConsistencyCircuitMultiTree<F>,
            FSMultiSetCircuitMultiTree<F>,
            InputMultiSetCircuitMultiTree<F>,
            BinaryRecompCircuitMultiTree<F>,
            PathCheckCircuitBatchedMul<F>,
            BinDecomp4BitIsBinaryCircuitBatchedMultiTree<F>,
            BinDecomp16BitIsBinaryCircuitBatchedMultiTree<F>,
            BinDecomp16BitIsBinaryCircuitMultiTree<F>,
            BinDecomp16BitIsBinaryCircuitMultiTree<F>,
            Vec<InputLayerEnum<F, PoseidonTranscript<F>>>, // input layers, including random layers
            Vec<CommitmentEnum<F>>,                        // input layers' commitments
        ),
        GKRError,
    > {
        let (
            mut input_samples_mle_vecs,
            mut permuted_input_samples_mle_vecs,
            mut decision_node_paths_mle_vecs,
            mut leaf_node_paths_mle_vecs,
            mut multiplicities_bin_decomp_mle_decision_vec,
            mut multiplicities_bin_decomp_mle_leaf_vec,
            mut decision_nodes_mle_vec,
            mut leaf_nodes_mle_vec,
            mut binary_decomp_diffs_mle_vecs,
            mut multiplicities_bin_decomp_mle_input_vecs
        ): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, )
         = self.batched_zkdt_circuit_mles_tree.clone().into_iter()
                .map(|batched_mle| (
                    batched_mle.input_samples_mle_vec,
                    batched_mle.permuted_input_samples_mle_vec,
                    batched_mle.decision_node_paths_mle_vec,
                    batched_mle.leaf_node_paths_mle_vec,
                    batched_mle.multiplicities_bin_decomp_mle_decision,
                    batched_mle.multiplicities_bin_decomp_mle_leaf,
                    batched_mle.decision_nodes_mle,
                    batched_mle.leaf_nodes_mle,
                    batched_mle.binary_decomp_diffs_mle_vec,
                    batched_mle.multiplicities_bin_decomp_mle_input_vec,
                ))
                .multiunzip();

        // deal w input
        let input_samples_mle_combined_vec =
            input_samples_mle_vecs.iter().map( 
                |input_samples_mle_vec| {
                    DenseMle::<F, InputAttribute<F>>::combine_mle_batch(input_samples_mle_vec.clone())
            }).collect_vec();
        let mut input_samples_mle_combined = DenseMle::<F, F>::combine_mle_batch(input_samples_mle_combined_vec);

        let permuted_input_samples_mle_vec_combined_vec = permuted_input_samples_mle_vecs.iter().map(
            |permuted_input_samples_mle_vec| {
                DenseMle::<F, InputAttribute<F>>::combine_mle_batch(
                    permuted_input_samples_mle_vec.clone(),
                )
            }).collect_vec();
        let mut permuted_input_samples_mle_vec_combined = DenseMle::<F, F>::combine_mle_batch(permuted_input_samples_mle_vec_combined_vec);
                
        let decision_node_paths_mle_vec_combined_vec = decision_node_paths_mle_vecs.iter().map(
            |decision_node_paths_mle_vec| {
                DenseMle::<F, DecisionNode<F>>::combine_mle_batch(decision_node_paths_mle_vec.clone())
            }).collect_vec();
        let mut decision_node_paths_mle_vec_combined = DenseMle::<F, F>::combine_mle_batch(decision_node_paths_mle_vec_combined_vec);

        let leaf_node_paths_mle_vec_combined_vec = leaf_node_paths_mle_vecs.iter().map(
            |leaf_node_paths_mle_vec| {
                DenseMle::<F, LeafNode<F>>::combine_mle_batch(leaf_node_paths_mle_vec.clone())
            }
        ).collect_vec();
        let mut leaf_node_paths_mle_vec_combined = DenseMle::<F, F>::combine_mle_batch(leaf_node_paths_mle_vec_combined_vec);
 
        let combined_batched_diff_signed_bin_decomp_mle_vec = binary_decomp_diffs_mle_vecs.iter().map(
            |binary_decomp_diffs_mle_vec| {
                DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(
                    binary_decomp_diffs_mle_vec.clone(),
                )
            }
        ).collect_vec();
        let mut combined_batched_diff_signed_bin_decomp_mle = DenseMle::<F, F>::combine_mle_batch(combined_batched_diff_signed_bin_decomp_mle_vec);

        let multiplicities_bin_decomp_mle_input_vec_combined_vec = multiplicities_bin_decomp_mle_input_vecs.iter().map(
            |multiplicities_bin_decomp_mle_input_vec| {
                DenseMle::<F, BinDecomp4Bit<F>>::combine_mle_batch(
                    multiplicities_bin_decomp_mle_input_vec.clone(),
                )
            }
        ).collect_vec();
        let mut multiplicities_bin_decomp_mle_input_vec_combined = DenseMle::<F, F>::combine_mle_batch(multiplicities_bin_decomp_mle_input_vec_combined_vec);


        let mut multiplicities_bin_decomp_mle_decision_combined = DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(multiplicities_bin_decomp_mle_decision_vec.clone());
        let mut multiplicities_bin_decomp_mle_leaf_combined = DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(multiplicities_bin_decomp_mle_leaf_vec.clone());
        let mut decision_nodes_mle_combined = DenseMle::<F, DecisionNode<F>>::combine_mle_batch(decision_nodes_mle_vec.clone());
        let mut leaf_nodes_mle_combined = DenseMle::<F, LeafNode<F>>::combine_mle_batch(leaf_nodes_mle_vec.clone());

        // Input layer shenanigans -- we need the following:
        // a) Precommitted Ligero input layer for tree itself (LayerId: 0)
        // b) Ligero input layer for just the inputs themselves (LayerId: 1)
        // c) Ligero input layer for all the auxiliaries (LayerId: 2)
        // d) Public input layer for all the leaf node outputs (LayerId: 3)
        // e) FS-style input layer for all the random packing constants + challenges (LayerId: 4, 5, 6)
        // TODO!(ryancao): Make it so that we don't have to manually assign all of the layer IDs for input layer MLEs...

        // --- Input layer 0 ---
        decision_nodes_mle_combined.layer_id = LayerId::Input(0);
        leaf_nodes_mle_combined.layer_id = LayerId::Input(0);
        let tree_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut decision_nodes_mle_combined),
            Box::new(&mut leaf_nodes_mle_combined),
        ];

        decision_nodes_mle_vec.iter_mut().for_each(
            |mle| mle.layer_id = LayerId::Input(2)
        );

        leaf_nodes_mle_vec.iter_mut().for_each(
            |mle| mle.layer_id = LayerId::Input(2)
        );


        // --- Input layer 1 ---
        input_samples_mle_combined.layer_id = LayerId::Input(1);
        input_samples_mle_vecs.iter_mut().for_each(|mle_vec| {
            mle_vec.iter_mut().for_each (|mle| {
                mle.layer_id = LayerId::Input(1);
            })
        });
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut input_samples_mle_combined)];

        // --- Input layer 2 ---
        permuted_input_samples_mle_vec_combined.layer_id = LayerId::Input(2);
        decision_node_paths_mle_vec_combined.layer_id = LayerId::Input(2);
        multiplicities_bin_decomp_mle_decision_combined.layer_id = LayerId::Input(2);
        multiplicities_bin_decomp_mle_leaf_combined.layer_id = LayerId::Input(2);
        combined_batched_diff_signed_bin_decomp_mle.layer_id = LayerId::Input(2);
        multiplicities_bin_decomp_mle_input_vec_combined.layer_id = LayerId::Input(2);


        permuted_input_samples_mle_vecs.iter_mut().for_each(|mle_vec| {
            mle_vec.iter_mut().for_each(|mle| {
                mle.layer_id = LayerId::Input(2);
            })
        });
        decision_node_paths_mle_vecs.iter_mut().for_each(|mle_vec| {
            mle_vec.iter_mut().for_each(|mle| {
                mle.layer_id = LayerId::Input(2);
            })
        });
        binary_decomp_diffs_mle_vecs.iter_mut().for_each(|mle_vec| {
            mle_vec.iter_mut().for_each(|mle| {
                mle.layer_id = LayerId::Input(2);
            })
        });
        multiplicities_bin_decomp_mle_input_vecs
            .iter_mut()
            .for_each(|mle_vec| {
                mle_vec.iter_mut().for_each(|mle| {
                    mle.layer_id = LayerId::Input(2);
                })
            });
        
        multiplicities_bin_decomp_mle_decision_vec.iter_mut().for_each(
            |mle| mle.layer_id = LayerId::Input(2)
        );

        multiplicities_bin_decomp_mle_leaf_vec.iter_mut().for_each(
            |mle| mle.layer_id = LayerId::Input(2)
        );


        let aux_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut permuted_input_samples_mle_vec_combined),
            Box::new(&mut decision_node_paths_mle_vec_combined),
            Box::new(&mut multiplicities_bin_decomp_mle_decision_combined),
            Box::new(&mut multiplicities_bin_decomp_mle_leaf_combined),
            Box::new(&mut combined_batched_diff_signed_bin_decomp_mle),
            Box::new(&mut multiplicities_bin_decomp_mle_input_vec_combined),
        ];

        // --- Input layer 3 ---
        leaf_node_paths_mle_vec_combined.layer_id = LayerId::Input(3);
        leaf_node_paths_mle_vecs.iter_mut().for_each(|mle_vec| {
            mle_vec.iter_mut().for_each(|mle| {
                mle.layer_id = LayerId::Input(3);
            })
        });
        let public_path_leaf_node_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut leaf_node_paths_mle_vec_combined)];

        // --- a) Precommitted Ligero input layer for tree itself (LayerId: 0) ---
        let tree_mle_input_layer_builder =
            InputLayerBuilder::new(tree_mles, None, LayerId::Input(0));

        // --- b) Ligero input layer for just the inputs themselves (LayerId: 1) ---
        let input_mles_input_layer_builder =
            InputLayerBuilder::new(input_mles, None, LayerId::Input(1));

        // --- c) Ligero input layer for all the auxiliaries (LayerId: 2) ---
        let aux_mles_input_layer_builder =
            InputLayerBuilder::new(aux_mles, None, LayerId::Input(2));

        // --- d) Public input layer for the path leaf nodes (LayerId: 3) ---
        let public_path_leaf_node_mles_input_layer_builder =
            InputLayerBuilder::new(public_path_leaf_node_mles, None, LayerId::Input(3));

        // --- Convert all the input layer builders into input layers ---
        let (_ligero_encoding, tree_ligero_commit, tree_ligero_root, tree_ligero_aux): (
            LigeroEncoding<F>,
            LcCommit<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>,
            LcRoot<LigeroEncoding<F>, F>,
            LcProofAuxiliaryInfo,
        ) = {
            let mut file = std::fs::File::open(&self.tree_precommit_filepath).unwrap();
            let initial_buffer_size = file.metadata().map(|m| m.len() as usize + 1).unwrap_or(0);
            let mut bufreader = Vec::with_capacity(initial_buffer_size);
            file.read_to_end(&mut bufreader).unwrap();
            serde_json::de::from_slice(&bufreader[..]).unwrap()
        };
        let tree_mle_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> =
            tree_mle_input_layer_builder.to_input_layer_with_precommit(
                tree_ligero_commit,
                tree_ligero_aux,
                tree_ligero_root,
            );

        let (
            _ligero_encoding,
            sample_minibatch_ligero_commit,
            sample_minibatch_ligero_root,
            sample_minibatch_ligero_aux,
        ): (
            LigeroEncoding<F>,
            LcCommit<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>,
            LcRoot<LigeroEncoding<F>, F>,
            LcProofAuxiliaryInfo,
        ) = {
            // METHOD 0: Use no buffer at all.
            // let file = std::fs::File::open(&self.sample_minibatch_precommit_filepath).unwrap();
            // let res = from_reader(&file).unwrap();

            // METHOD 1: Read everything into the buffer.
            let mut file = std::fs::File::open(&self.sample_minibatch_precommit_filepath).unwrap();
            let initial_buffer_size = file.metadata().map(|m| m.len() as usize + 1).unwrap_or(0);
            let mut bufreader = Vec::with_capacity(initial_buffer_size);
            file.read_to_end(&mut bufreader).unwrap();
            let res = serde_json::de::from_slice(&bufreader[..]).unwrap();

            // METHOD 2: Use a buffer of a default size.
            // let file = std::fs::File::open(&self.sample_minibatch_precommit_filepath).unwrap();
            // let mut bufreader = BufReader::new(file);
            // let res = from_reader(&mut bufreader).unwrap();
            res
        };
        let input_mles_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> =
            input_mles_input_layer_builder.to_input_layer_with_precommit(
                sample_minibatch_ligero_commit,
                sample_minibatch_ligero_aux,
                sample_minibatch_ligero_root,
            );

        let aux_mles_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> =
            aux_mles_input_layer_builder.to_input_layer_with_rho_inv(
                self.rho_inv,
                self.ratio,
            );
        let public_path_leaf_node_mles_input_layer: PublicInputLayer<F, PoseidonTranscript<F>> =
            public_path_leaf_node_mles_input_layer_builder.to_input_layer();
        let mut tree_mle_input_layer = tree_mle_input_layer.to_enum();
        let mut input_mles_input_layer = input_mles_input_layer.to_enum();
        let mut aux_mles_input_layer = aux_mles_input_layer.to_enum();
        let mut public_path_leaf_node_mles_input_layer =
            public_path_leaf_node_mles_input_layer.to_enum();

        // --- Add input layer derived prefix bits to vectors ---

        // --- Zeroth input layer ---

        decision_nodes_mle_vec.iter_mut().for_each(
            |decision_nodes_mle| {
                decision_nodes_mle.set_prefix_bits(decision_nodes_mle_combined.get_prefix_bits())
            }
        );

        leaf_nodes_mle_vec.iter_mut().for_each(
            |leaf_nodes_mle| {
                leaf_nodes_mle.set_prefix_bits(leaf_nodes_mle_combined.get_prefix_bits())
            }
        );


        // --- First input layer ---
        
        decision_node_paths_mle_vecs.iter_mut().for_each(
            |decision_node_paths_mle_vec| {
                decision_node_paths_mle_vec.iter_mut().for_each(|decision_node_paths_mle| {
                    decision_node_paths_mle.set_prefix_bits(decision_node_paths_mle_vec_combined.get_prefix_bits());
                });
            });

        leaf_node_paths_mle_vecs.iter_mut().for_each(
            |leaf_node_paths_mle_vec| {
                leaf_node_paths_mle_vec.iter_mut().for_each(|leaf_node_paths_mle| {
                    leaf_node_paths_mle.set_prefix_bits(leaf_node_paths_mle_vec_combined.get_prefix_bits());
                });
            });

        // --- Second input layer ---

        input_samples_mle_vecs.iter_mut().for_each(
            |input_samples_mle_vec| {
                input_samples_mle_vec.iter_mut().for_each( |input_samples_mle| {
                    input_samples_mle.set_prefix_bits(input_samples_mle_combined.get_prefix_bits());
                });
            });

        // --- Third input layer ---

        permuted_input_samples_mle_vecs.iter_mut().for_each(|permuted_input_samples_mle_vec| {
            permuted_input_samples_mle_vec.iter_mut().for_each(|permuted_input_samples_mle| {
                permuted_input_samples_mle.set_prefix_bits(permuted_input_samples_mle_vec_combined.get_prefix_bits());
            });
        });

        decision_node_paths_mle_vecs.iter_mut().for_each(|decision_node_paths_mle_vec| {
            decision_node_paths_mle_vec.iter_mut().for_each(|decision_node_paths_mle| {
                decision_node_paths_mle.set_prefix_bits(decision_node_paths_mle_vec_combined.get_prefix_bits());
            });
        });

        binary_decomp_diffs_mle_vecs.iter_mut().for_each(|binary_decomp_diffs_mle_vec| {
            binary_decomp_diffs_mle_vec.iter_mut().for_each(|binary_decomp_diffs_mle| {
                binary_decomp_diffs_mle.set_prefix_bits(binary_decomp_diffs_mle.get_prefix_bits())
            });
        });

        multiplicities_bin_decomp_mle_input_vecs.iter_mut().for_each(|multiplicities_bin_decomp_mle_input_vec| {
            multiplicities_bin_decomp_mle_input_vec.iter_mut().for_each(|multiplicities_bin_decomp_mle_input| {
                multiplicities_bin_decomp_mle_input.set_prefix_bits(multiplicities_bin_decomp_mle_decision_combined.get_prefix_bits())
            });
        });

        multiplicities_bin_decomp_mle_decision_vec.iter_mut().for_each(|multiplicities_bin_decomp_mle_decision| {
            multiplicities_bin_decomp_mle_decision.set_prefix_bits(multiplicities_bin_decomp_mle_decision_combined.get_prefix_bits())
        });

        multiplicities_bin_decomp_mle_leaf_vec.iter_mut().for_each(|multiplicities_bin_decomp_mle_leaf| {
            multiplicities_bin_decomp_mle_leaf.set_prefix_bits(multiplicities_bin_decomp_mle_leaf_combined.get_prefix_bits())
        });

        // --- Last input layer ---
        leaf_node_paths_mle_vecs.iter_mut().for_each(|leaf_node_paths_mle_vec| {
            leaf_node_paths_mle_vec.iter_mut().for_each(|leaf_node_paths_mle| {
                leaf_node_paths_mle.set_prefix_bits(leaf_node_paths_mle_vec_combined.get_prefix_bits())
            });
        });

        // --- Add commitments to transcript so they are taken into account before the FS input layers are sampled ---
        let tree_mle_commit = tree_mle_input_layer
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;
        InputLayerEnum::append_commitment_to_transcript(&tree_mle_commit, transcript).unwrap();

        let input_mle_commit = input_mles_input_layer
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;
        InputLayerEnum::append_commitment_to_transcript(&input_mle_commit, transcript).unwrap();

        let aux_mle_commit = aux_mles_input_layer
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;
        InputLayerEnum::append_commitment_to_transcript(&aux_mle_commit, transcript).unwrap();

        let public_path_leaf_node_mle_commit = public_path_leaf_node_mles_input_layer
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;
        InputLayerEnum::append_commitment_to_transcript(
            &public_path_leaf_node_mle_commit,
            transcript,
        )
        .unwrap();

        // --- FS layers must also have LayerId::Input(.)s! ---
        // Input(4)
        let random_r = RandomInputLayer::new(transcript, 1, LayerId::Input(4));
        let r_mle = random_r.get_mle();
        let mut random_r = random_r.to_enum();
        let random_r_commit = random_r.commit().map_err(GKRError::InputLayerError)?;

        // Input(5)
        let random_r_packing = RandomInputLayer::new(transcript, 1, LayerId::Input(5));
        let r_packing_mle = random_r_packing.get_mle();
        let mut random_r_packing = random_r_packing.to_enum();
        let random_r_packing_commit = random_r_packing
            .commit()
            .map_err(GKRError::InputLayerError)?;

        // Input(6)
        let random_r_packing_another = RandomInputLayer::new(transcript, 1, LayerId::Input(6));
        let r_packing_another_mle = random_r_packing_another.get_mle();
        let mut random_r_packing_another = random_r_packing_another.to_enum();
        let random_r_packing_another_commit = random_r_packing_another
            .commit()
            .map_err(GKRError::InputLayerError)?;

        // --- Construct the actual circuit structs ---
        let attribute_consistency_circuit = AttributeConsistencyCircuitMultiTree {
            permuted_input_data_mle_trees_vec: permuted_input_samples_mle_vecs.clone(),
            decision_node_paths_mle_trees_vec: decision_node_paths_mle_vecs.clone(),
        };

        // --- TODO!(% Labs): Get rid of all the `.clone()`s ---
        let multiset_circuit = FSMultiSetCircuitMultiTree {
            decision_nodes_mle_tree: decision_nodes_mle_vec,
            leaf_nodes_mle_tree: leaf_nodes_mle_vec,
            r_mle: r_mle.clone(),
            r_packing_mle: r_packing_mle.clone(),
            r_packing_another_mle,
            multiplicities_bin_decomp_mle_decision_tree: multiplicities_bin_decomp_mle_decision_vec.clone(),
            multiplicities_bin_decomp_mle_leaf_tree: multiplicities_bin_decomp_mle_leaf_vec.clone(),
            decision_node_paths_mle_vec_tree: decision_node_paths_mle_vecs.clone(),
            leaf_node_paths_mle_vec_tree: leaf_node_paths_mle_vecs.clone(),
        };

        let input_multiset_circuit = InputMultiSetCircuitMultiTree {
            r_mle,
            r_packing_mle,
            input_data_mle_vec_tree: input_samples_mle_vecs,
            permuted_input_data_mle_vec_tree: permuted_input_samples_mle_vecs.clone(),
            multiplicities_bin_decomp_mle_input_vec_tree: multiplicities_bin_decomp_mle_input_vecs.clone(),
        };


        let binary_recomp_circuit_batched = BinaryRecompCircuitMultiTree::new(
            decision_node_paths_mle_vecs.clone(),
            permuted_input_samples_mle_vecs.clone(),
            binary_decomp_diffs_mle_vecs.clone(),
        );

        let path_consistency_circuit_batched = PathCheckCircuitBatchedMul::new(
            decision_node_paths_mle_vecs[0].clone(),
            leaf_node_paths_mle_vecs[0].clone(),
            binary_decomp_diffs_mle_vecs[0].clone(),
        );

        // --- Bits are binary check for each of the binary decomposition input MLEs ---
        let bits_binary_16_bit_batched =
            BinDecomp16BitIsBinaryCircuitBatchedMultiTree::new(binary_decomp_diffs_mle_vecs);

        let bits_binary_4_bit_batched =
            BinDecomp4BitIsBinaryCircuitBatchedMultiTree::new(multiplicities_bin_decomp_mle_input_vecs);

        let bits_are_binary_multiset_decision_circuit =
            BinDecomp16BitIsBinaryCircuitMultiTree::new(multiplicities_bin_decomp_mle_decision_vec);
        let bits_are_binary_multiset_leaf_circuit =
            BinDecomp16BitIsBinaryCircuitMultiTree::new(multiplicities_bin_decomp_mle_leaf_vec);

        Ok((
            // --- Actual circuit components ---
            attribute_consistency_circuit,
            multiset_circuit,
            input_multiset_circuit,
            binary_recomp_circuit_batched,
            path_consistency_circuit_batched,
            bits_binary_4_bit_batched,
            bits_binary_16_bit_batched,
            bits_are_binary_multiset_decision_circuit,
            bits_are_binary_multiset_leaf_circuit,
            // --- Input layers ---
            vec![
                // tree_mle_input_layer,
                // input_mles_input_layer,
                aux_mles_input_layer,
                public_path_leaf_node_mles_input_layer,
                random_r,
                random_r_packing,
                random_r_packing_another,
            ],
            vec![
                // tree_mle_commit,
                // input_mle_commit,
                aux_mle_commit,
                public_path_leaf_node_mle_commit,
                random_r_commit,
                random_r_packing_commit,
                random_r_packing_another_commit,
            ],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::ZKDTMultiTreeCircuit;
    use crate::zkdt::input_data_to_circuit_adapter::{load_upshot_data_single_tree_batch, convert_zkdt_circuit_data_into_mles, MinibatchData};
    use crate::{prover::tests::test_circuit, zkdt::input_data_to_circuit_adapter::BatchedZKDTCircuitMles};
    use crate::zkdt::cache_upshot_catboost_inputs_for_testing::generate_mles_batch_catboost_single_tree;
    use ark_std::{end_timer, start_timer};
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use itertools::Itertools;
    use std::path::Path;

    use chrono;
    use log::LevelFilter;
    use std::io::Write;

    #[test]
    fn test_zkdt_2_tree_circuit() {
        env_logger::Builder::new()
            .format(|buf, record| {
                writeln!(
                    buf,
                    "----> {}:{} {} [{}]:\n{}",
                    record.file().unwrap_or("unknown"),
                    record.line().unwrap_or(0),
                    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S"),
                    record.level(),
                    record.args()
                )
            })
            .filter(None, LevelFilter::Error)
            .init();

        let minibatch_data = MinibatchData {log_sample_minibatch_size: 3, sample_minibatch_number: 2};
        let trees_batched_data: Vec<BatchedZKDTCircuitMles<Fr>> = (0..2).map(
            |tree_num| {
                // --- Read in the Upshot data from file ---
                let (zkdt_circuit_data, (tree_height, input_len), _) =
                load_upshot_data_single_tree_batch::<Fr>(
                    Some(minibatch_data.clone()),
                    tree_num,
                    Path::new(&"upshot_data/quantized-upshot-model.json".to_string()),
                    Path::new(&"upshot_data/upshot-quantized-samples.npy".to_string()),
                );
                let (batched_catboost_mles, (_, _)) =
                    convert_zkdt_circuit_data_into_mles(zkdt_circuit_data, tree_height, input_len);
                batched_catboost_mles
        }).collect_vec();
        
        let combined_circuit = ZKDTMultiTreeCircuit {
            batched_zkdt_circuit_mles_tree: trees_batched_data,
            tree_precommit_filepath: "upshot_data/tree_ligero_commitments/tree_commitment_0.json".to_string(),
            sample_minibatch_precommit_filepath: "upshot_data/sample_minibatch_commitments/sample_minibatch_logsize_10_commitment_0.json".to_string(),
            rho_inv: 4,
            ratio: 1_f64,
        };

        test_circuit(
            combined_circuit,
            None,
        );
    }

    // #[test]
    // fn bench_zkdt_circuits() {
    //     (10..11).for_each(|batch_size| {
    //         let circuit_timer = start_timer!(|| format!("zkdt circuit, batch_size 2^{batch_size}"));
    //         let wit_gen_timer = start_timer!(|| "wit gen");

    //         let (batched_zkdt_circuit_mles, (_, _)) = generate_mles_batch_catboost_single_tree::<Fr>(
    //             batch_size,
    //             Path::new("upshot_data/"),
    //         );
    //         end_timer!(wit_gen_timer);

    //         let combined_circuit = ZKDTMultiTreeCircuit {
    //             batched_zkdt_circuit_mles,
    //             tree_precommit_filepath:
    //                 "upshot_data/tree_ligero_commitments/tree_commitment_0.json".to_string(),
    //             sample_minibatch_precommit_filepath:
    //                 "upshot_data/sample_minibatch_commitments/sample_minibatch_logsize_10_commitment_0.json".to_string(),
    //             rho_inv: 4,
    //             ratio: 1_f64,
    //         };

    //         test_circuit(combined_circuit, None);
    //     });
    // }
}