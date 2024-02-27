// use ark_std::log2;
// use itertools::{repeat_n, Itertools};
// use remainder::{layer::{batched::{combine_zero_mle_ref, BatchedLayer}, LayerId}, mle::{dense::DenseMle, structs::BinDecomp16Bit, Mle, MleIndex, MleRef}, prover::{input_layer::{combine_input_layers::InputLayerBuilder, ligero_input_layer::LigeroInputLayer, InputLayer}, GKRCircuit, Layers, Witness}};
// use remainder_shared_types::{transcript::{poseidon_transcript::PoseidonTranscript, Transcript}, FieldExt, Fr};

// use crate::{circuit_builders::{PositiveBinaryRecompBuilder, BinaryRecompCheckerBuilder}, utils::{generate_16_bit_decomp, generate_16_bit_decomp_signed}};

// /// Batched version of the binary recomposition circuit (see below)!
// pub struct BinaryRecompCircuit<F: FieldExt> {
//     mle: DenseMle<F, F>,
//     signed_bin_decomp_mle: DenseMle<F, BinDecomp16Bit<F>>,
// }

// impl<F: FieldExt> GKRCircuit<F> for BinaryRecompCircuit<F> {
//     type Transcript = PoseidonTranscript<F>;

//     fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

//         // --- For the input layer, we need to first merge all of the input MLEs FIRST by mle_idx, then by dataparallel index ---
//         // --- This assures that (going left-to-right in terms of the bits) we have [input_prefix_bits], [dataparallel_bits], [mle_idx], [iterated_bits] ---
//         // let mut combined_mles = DenseMle::<F, F>::combine_mle_batch(self.mles.clone());
//         // let mut combined_signed_bin_decomp_mles = DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(self.signed_bin_decomp_mles.clone());

//         // --- Inputs to the circuit are just these three MLEs ---
//         let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle), Box::new(&mut self.signed_bin_decomp_mle)];
//         let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));

//         // --- Create `Layers` struct to add layers to ---
//         let mut layers: Layers<F, Self::Transcript> = Layers::new();

//         // --- First we create the positive binary recomp builder ---
//         let bin_recomp_builder = PositiveBinaryRecompBuilder::new(self.signed_bin_decomp_mle.clone());
//         let pos_bin_recomp_mle = layers.add_gkr(bin_recomp_builder);

//         // --- Finally, the recomp checker ---
//         let recomp_checker_builder = 

//         let batched_recomp_checker_builder = BatchedLayer::new(
//             self.signed_bin_decomp_mle.iter_mut().zip(
//                 pos_bin_recomp_mle.into_iter().zip(
//                     self.mle.clone().into_iter()
//                 )
//             )
//             .map(|(signed_bit_decomp_mle, (pos_bin_recomp_mle, raw_diff_mle))| {

//                 BinaryRecompCheckerBuilder::new(
//                     raw_diff_mle,
//                     signed_bit_decomp_mle.clone(),
//                     pos_bin_recomp_mle,
//                 )
//             }
//         ).collect_vec());

//         // --- Grab output layer and flatten ---
//         let batched_recomp_checker_result_mle = layers.add_gkr(batched_recomp_checker_builder);
//         let flattened_batched_recomp_checker_result_mle = combine_zero_mle_ref(batched_recomp_checker_result_mle);

//         // --- Create input layers ---
//         let live_committed_input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer_with_rho_inv(4_u8, 1_f64);

//         Witness { layers, output_layers: vec![flattened_batched_recomp_checker_result_mle.get_enum()], input_layers: vec![live_committed_input_layer.to_enum()] }
//     }
// }

// impl<F: FieldExt> BinaryRecompCircuit<F> {
//     /// Creates a new instance of BinaryRecompCircuit
//     pub fn new(
//         mles: Vec<DenseMle<F, F>>,
//         signed_bin_decomp_mles: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
//     ) -> Self {
//         Self {
//             mle: mles,
//             signed_bin_decomp_mle: signed_bin_decomp_mles,
//         }
//     }
// }

// #[test]
// fn test_bin_recomp_16_bits() {

//     // generate a random 16 bits, compute the number from the decomposition
//     let (
//         binary_decomp_mle_vec,
//         binary_recomp_mle_vec
//     ) = generate_16_bit_decomp_signed::<Fr>(2);

//     let mut circuit = BinaryRecompCircuit::new(
//         binary_recomp_mle_vec,
//         binary_decomp_mle_vec,
//     );

//     let mut transcript = PoseidonTranscript::new("Bin Decomp Bits Are Binary Transcript");
//     let proof = circuit.prove(&mut transcript);

//     match proof {
//         Ok(proof) => {
//             let mut transcript = PoseidonTranscript::new("Bin Decomp Bits Are Binary Transcript");
//             let result = circuit.verify(&mut transcript, proof);
//             if let Err(err) = result {
//                 println!("{}", err);
//                 panic!();
//             }
//         },
//         Err(err) => {
//             println!("{}", err);
//             panic!();
//         }
//     }
// }