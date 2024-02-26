use ark_std::log2;
use itertools::{repeat_n, Itertools};
use remainder::{layer::{batched::{combine_zero_mle_ref, BatchedLayer}, LayerId}, mle::{dense::DenseMle, structs::BinDecomp16Bit, Mle, MleIndex, MleRef}, prover::{input_layer::{combine_input_layers::InputLayerBuilder, ligero_input_layer::LigeroInputLayer, InputLayer}, GKRCircuit, Layers, Witness}};
use remainder_shared_types::{transcript::{poseidon_transcript::PoseidonTranscript, Transcript}, FieldExt, Fr};

use crate::{circuit_builders::{BinaryRecompBuilder, BinaryRecompCheckerBuilder}, utils::generate_16_bit_decomp};

/// Batched version of the binary recomposition circuit (see below)!
pub struct BinaryRecompCircuit<F: FieldExt> {
    mles: Vec<DenseMle<F, F>>,
    signed_bin_decomp_mles: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinaryRecompCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- For the input layer, we need to first merge all of the input MLEs FIRST by mle_idx, then by dataparallel index ---
        // --- This assures that (going left-to-right in terms of the bits) we have [input_prefix_bits], [dataparallel_bits], [mle_idx], [iterated_bits] ---
        let mut combined_mles = DenseMle::<F, F>::combine_mle_batch(self.mles.clone());
        let mut combined_signed_bin_decomp_mles = DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(self.signed_bin_decomp_mles.clone());

        // --- Inputs to the circuit are just these three MLEs ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut combined_mles), Box::new(&mut combined_signed_bin_decomp_mles)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));

        // --- Dataparallel/batching stuff + sanitychecks ---
        let num_subcircuit_copies = self.signed_bin_decomp_mles.len();
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- First we create the positive binary recomp builder ---
        let pos_bin_recomp_builders = self.signed_bin_decomp_mles.iter_mut().map(|signed_bin_decomp_mle| {
            // --- Prefix bits should be [input_prefix_bits], [dataparallel_bits] ---
            // TODO!(ryancao): Note that strictly speaking we shouldn't be adding dataparallel bits but need to for
            // now for a specific batching scenario
            signed_bin_decomp_mle.set_prefix_bits(
                Some(
                    combined_signed_bin_decomp_mles.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                    ).collect_vec()
                )
            );
            BinaryRecompBuilder::new(signed_bin_decomp_mle.clone())
        }).collect();

        let batched_bin_recomp_builder = BatchedLayer::new(pos_bin_recomp_builders);


        for mle in self.mles.iter_mut() {
            mle.set_prefix_bits(Some(
                combined_mles.get_prefix_bits().iter().flatten().cloned().chain(
                    repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                ).collect_vec()
            ));
        }

        let batched_pos_bin_recomp_mle = layers.add_gkr(batched_bin_recomp_builder);

        // --- Finally, the recomp checker ---
        let batched_recomp_checker_builder = BatchedLayer::new(
            self.signed_bin_decomp_mles.iter_mut().zip(
                batched_pos_bin_recomp_mle.into_iter().zip(
                    self.mles.clone().into_iter()
                )
            )
            .map(|(signed_bit_decomp_mle, (pos_bin_recomp_mle, raw_diff_mle))| {

                // // do we need to add prefix bits again here??
                // // --- Add prefix bits to the thing which was indexed earlier ---
                // signed_bit_decomp_mle.set_prefix_bits(
                //     Some(
                //         combined_signed_bin_decomp_mles.get_prefix_bits().iter().flatten().cloned().chain(
                //             repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                //         ).collect_vec()
                //     )
                // );

                BinaryRecompCheckerBuilder::new(
                    raw_diff_mle,
                    signed_bit_decomp_mle.clone(),
                    pos_bin_recomp_mle,
                )
            }
        ).collect_vec());

        // --- Grab output layer and flatten ---
        let batched_recomp_checker_result_mle = layers.add_gkr(batched_recomp_checker_builder);
        let flattened_batched_recomp_checker_result_mle = combine_zero_mle_ref(batched_recomp_checker_result_mle);

        // --- Create input layers ---
        let live_committed_input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer_with_rho_inv(4_u8, 1_f64);

        Witness { layers, output_layers: vec![flattened_batched_recomp_checker_result_mle.get_enum()], input_layers: vec![live_committed_input_layer.to_enum()] }
    }
}

#[test]
fn test_bin_recomp_16_bits() {

    // generate a random 16 bits, compute the number from the decomposition
    let (
        binary_decomp_mle_vec,
        binary_recomp_mle_vec
    ) = generate_16_bit_decomp::<Fr>(4, 2);

    let mut circuit = BinaryRecompCircuit::new(
        binary_recomp_mle_vec,
        binary_decomp_mle_vec,
    );

    let mut transcript = PoseidonTranscript::new("Bin Decomp Bits Are Binary Transcript");
    let proof = circuit.prove(&mut transcript);

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


impl<F: FieldExt> BinaryRecompCircuit<F> {
    /// Creates a new instance of BinaryRecompCircuit
    pub fn new(
        mles: Vec<DenseMle<F, F>>,
        signed_bin_decomp_mles: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    ) -> Self {
        Self {
            mles,
            signed_bin_decomp_mles,
        }
    }

    // /// This does exactly the same thing as `synthesize()` above, but
    // /// takes in prefix bits for each of the input layer MLEs as opposed
    // /// to synthesizing its own input layer.
    // pub fn yield_sub_circuit(&mut self) -> Witness<F, <BinaryRecompCircuit<F> as GKRCircuit<F>>::Transcript> {

    //     // --- NOTE: There is no input layer creation, since this gets handled in the large circuit ---

    //     // --- Dataparallel/batching stuff + sanitychecks ---
    //     let num_subcircuit_copies = self.batched_decision_node_path_mle.len();
    //     let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;
    //     debug_assert_eq!(num_dataparallel_bits, log2(self.batched_permuted_inputs_mle.len()) as usize);
    //     debug_assert_eq!(num_dataparallel_bits, log2(self.batched_diff_signed_bin_decomp_mle.len()) as usize);

    //     // --- Create `Layers` struct to add layers to ---
    //     let mut layers: Layers<F, <BinaryRecompCircuit<F> as GKRCircuit<F>>::Transcript> = Layers::new();

    //     let batched_diff_signed_bin_decomp_mle_prefix_bits = self.batched_diff_signed_bin_decomp_mle[0].get_prefix_bits();
    //     // --- First we create the positive binary recomp builder ---
    //     let pos_bin_recomp_builders = self.batched_diff_signed_bin_decomp_mle.iter_mut().map(
    //         |signed_bit_decomp_mle| {
    //         // --- Prefix bits should be [input_prefix_bits], [dataparallel_bits] ---
    //         // TODO!(ryancao): Note that strictly speaking we shouldn't be adding dataparallel bits but need to for
    //         // now for a specific batching scenario
    //         signed_bit_decomp_mle.set_prefix_bits(
    //             Some(
    //                 signed_bit_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
    //                     repeat_n(MleIndex::Iterated, num_dataparallel_bits)
    //                 ).collect_vec()
    //             )
    //         );
    //         BinaryRecompBuilder::new(signed_bit_decomp_mle.clone())
    //     }).collect();

    //     let batched_bin_recomp_builder = BatchedLayer::new(pos_bin_recomp_builders);

    //     // --- Next, we create the diff builder ---
    //     let batched_diff_builder = BatchedLayer::new(
    //         self.batched_decision_node_path_mle.iter_mut().zip(
    //             self.batched_permuted_inputs_mle.iter_mut()
    //         ).map(|(decision_node_path_mle, permuted_inputs_mle)| {

    //             // --- Add prefix bits and batching bits to both (same comment as above) ---
    //             decision_node_path_mle.set_prefix_bits(Some(
    //                 decision_node_path_mle.get_prefix_bits().iter().flatten().cloned().chain(
    //                     repeat_n(MleIndex::Iterated, num_dataparallel_bits)
    //                 ).collect_vec()
    //             ));
    //             permuted_inputs_mle.set_prefix_bits(Some(
    //                 permuted_inputs_mle.get_prefix_bits().iter().flatten().cloned().chain(
    //                     repeat_n(MleIndex::Iterated, num_dataparallel_bits)
    //                 ).collect_vec()
    //             ));

    //         NodePathDiffBuilder::new(decision_node_path_mle.clone(), permuted_inputs_mle.clone())
    //     }).collect_vec());

    //     // --- Concatenate the previous two builders and add them to the circuit ---
    //     let builder = batched_bin_recomp_builder.concat(batched_diff_builder);
    //     let (batched_pos_bin_recomp_mle, batched_raw_diff_mle) = layers.add_gkr(builder);

    //     // --- Finally, the recomp checker ---
    //     let batched_recomp_checker_builder = BatchedLayer::new(
    //         self.batched_diff_signed_bin_decomp_mle.iter_mut().zip(
    //             batched_pos_bin_recomp_mle.into_iter().zip(
    //                 batched_raw_diff_mle.into_iter()
    //             )
    //         )
    //         .map(|(signed_bit_decomp_mle, (pos_bin_recomp_mle, raw_diff_mle))| {

    //             // --- Add prefix bits to the thing which was indexed earlier ---
    //             signed_bit_decomp_mle.set_prefix_bits(
    //                 Some(
    //                     batched_diff_signed_bin_decomp_mle_prefix_bits.iter().flatten().cloned().chain(
    //                         repeat_n(MleIndex::Iterated, num_dataparallel_bits)
    //                     ).collect_vec()
    //                 )
    //             );

    //             BinaryRecompCheckerBuilder::new(
    //                 raw_diff_mle,
    //                 signed_bit_decomp_mle.clone(),
    //                 pos_bin_recomp_mle,
    //             )
    //         }
    //     ).collect_vec());

    //     // --- Grab output layer and flatten ---
    //     let batched_recomp_checker_result_mle = layers.add_gkr(batched_recomp_checker_builder);
    //     let flattened_batched_recomp_checker_result_mle = combine_zero_mle_ref(batched_recomp_checker_result_mle);

    //     println!("# layers -- binary recomp: {:?}", layers.next_layer_id());

    //     Witness { layers, output_layers: vec![flattened_batched_recomp_checker_result_mle.get_enum()], input_layers: vec![] }
    // }
}