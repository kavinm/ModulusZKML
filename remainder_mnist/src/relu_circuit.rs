use ark_std::log2;
use itertools::{repeat_n, Itertools};
use remainder::{expression::ExpressionStandard, layer::{batched::{combine_zero_mle_ref, BatchedLayer}, from_mle, LayerId}, mle::{dense::DenseMle, structs::BinDecomp16Bit, zero::ZeroMleRef, Mle, MleIndex, MleRef}, prover::{input_layer::{combine_input_layers::InputLayerBuilder, ligero_input_layer::LigeroInputLayer, public_input_layer::PublicInputLayer, InputLayer}, GKRCircuit, Layers, Witness}};
use remainder_shared_types::{transcript::{poseidon_transcript::PoseidonTranscript, Transcript}, FieldExt, Fr};

use crate::{circuit_builders::{BinaryRecompCheckerBuilder, PositiveBinaryRecompBuilder}, utils::generate_16_bit_decomp_signed};


pub struct ReluCircuit<F: FieldExt> {
    pub mles: Vec<DenseMle<F, F>>,
    pub signed_bin_decomp_mles: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
}


impl<F: FieldExt> GKRCircuit<F> for ReluCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- For the input layer, we need to first merge all of the input MLEs FIRST by mle_idx, then by dataparallel index ---
        // --- This assures that (going left-to-right in terms of the bits) we have [input_prefix_bits], [dataparallel_bits], [mle_idx], [iterated_bits] ---
        let mut combined_mles = DenseMle::<F, F>::combine_mle_batch(self.mles.clone());
        let mut combined_signed_bin_decomp_mles = DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(self.signed_bin_decomp_mles.clone());

        // --- Dataparallel/batching stuff + sanitychecks ---
        let num_subcircuit_copies = self.signed_bin_decomp_mles.len();
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;
        debug_assert_eq!(num_dataparallel_bits, log2(self.mles.len()) as usize);

        // --- set the prefix bits with regard to the dataparallel_bits ---
        // --- Prefix bits should be [input_prefix_bits], [dataparallel_bits] ---
        for mle in self.mles.iter_mut() {
            mle.set_prefix_bits(Some(
                combined_mles.get_prefix_bits().iter().flatten().cloned().chain(
                    repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                ).collect_vec()
            ));
        }
        for signed_bin_decomp_mle in self.signed_bin_decomp_mles.iter_mut() {
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
        }

        // --- Inputs to the circuit are just these two MLEs ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut combined_mles), Box::new(&mut combined_signed_bin_decomp_mles)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));

        // --- Create input layers ---
        let live_committed_input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer_with_rho_inv(4_u8, 1_f64);

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // **************************** BEGIN: checking the bits recompute to the mles ****************************

        // --- First we create the positive binary recomp builder ---
        let pos_bin_recomp_builders = self.signed_bin_decomp_mles.iter_mut().map(|signed_bin_decomp_mle| {
            PositiveBinaryRecompBuilder::new(signed_bin_decomp_mle.clone())
        }).collect();

        let batched_bin_recomp_builder = BatchedLayer::new(pos_bin_recomp_builders);
        let batched_pos_bin_recomp_mle = layers.add_gkr(batched_bin_recomp_builder);

        // --- Finally, the recomp checker ---
        let recomp_checker_builders = BatchedLayer::new(
            self.signed_bin_decomp_mles.iter_mut().zip(
                batched_pos_bin_recomp_mle.into_iter().zip(
                    self.mles.clone().into_iter()
                )
            )
            .map(|(signed_bit_decomp_mle, (positive_recomp_mle, raw_diff_mle))| {

                BinaryRecompCheckerBuilder::new(
                    raw_diff_mle,
                    signed_bit_decomp_mle.clone(),
                    positive_recomp_mle,
                )
            }
        ).collect_vec());

        // --- Grab output layer and flatten ---
        let recomp_checker_results = combine_zero_mle_ref(layers.add_gkr(recomp_checker_builders));

        // **************************** END: checking the bits recompute to the mles ****************************

        // **************************** BEGIN: checking the bits are binary ****************************

        // --- Create the builders for (b_i)^2 - b_i ---
        let bits_are_binary_builders = self.signed_bin_decomp_mles.iter_mut().map(|signed_bin_decomp_mle| {
            
            from_mle(
                signed_bin_decomp_mle, 
                |signed_bin_decomp_mle| {
                    let combined_bin_decomp_mle_ref = signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
                    ExpressionStandard::Product(vec![combined_bin_decomp_mle_ref.clone(), combined_bin_decomp_mle_ref.clone()]) - ExpressionStandard::Mle(combined_bin_decomp_mle_ref)
                }, 
                |mle, id, prefix_bits| {
                    ZeroMleRef::new(mle.num_iterated_vars(), prefix_bits, id)
            })
        }).collect_vec();

        let bits_are_binary_results = combine_zero_mle_ref(layers.add_gkr(BatchedLayer::new(bits_are_binary_builders)));

        // **************************** END: checking the bits are binary ****************************

        Witness {
            layers,
            output_layers: vec![
                recomp_checker_results.get_enum(),
                bits_are_binary_results.get_enum(),
            ],
            input_layers: vec![live_committed_input_layer.to_enum()]
        }
    }
}


impl<F: FieldExt> ReluCircuit<F> {
    /// Creates a new instance of BinDecomp16BitsAreBinaryCircuit
    pub fn new(
        mles: Vec<DenseMle<F, F>>,
        signed_bin_decomp_mles: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    ) -> Self {
        Self {
            mles,
            signed_bin_decomp_mles,
        }
    }
}

#[test]
fn test_relu_circuit() {

    // generate a random 16 bits, compute the number from the decomposition
    let (
        binary_decomp_mle_vec,
        binary_recomp_mle_vec
    ) = generate_16_bit_decomp_signed::<Fr>(4, 2);

    let mut circuit = ReluCircuit::new(
        binary_recomp_mle_vec,
        binary_decomp_mle_vec,
    );

    let mut transcript = PoseidonTranscript::new("Relu Circuit Transcript");
    let proof = circuit.prove(&mut transcript);

    match proof {
        Ok(proof) => {
            let mut transcript = PoseidonTranscript::new("Relu Circuit Transcript");
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