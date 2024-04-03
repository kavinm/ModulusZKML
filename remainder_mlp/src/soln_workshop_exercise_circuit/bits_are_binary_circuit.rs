use ark_ff::Field;
use ark_std::{log2, test_rng};
use itertools::{repeat_n, Itertools};
use rand::Rng;
use remainder::{expression::ExpressionStandard, layer::{batched::{combine_zero_mle_ref, BatchedLayer}, from_mle, LayerId}, mle::{dense::DenseMle, structs::BinDecomp16Bit, zero::ZeroMleRef, Mle, MleIndex, MleRef}, prover::{input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer}, GKRCircuit, Layers, Witness}};
use remainder_shared_types::{transcript::{poseidon_transcript::PoseidonTranscript, Transcript}, FieldExt, Fr};

/// Checks that all of the bits within a `BinDecomp16Bit` are indeed binary
/// via b_i^2 - b_i = 0 (but it's batched)
pub struct BinDecomp16BitsAreBinaryCircuit<F: FieldExt> {
    signed_bin_decomp_mle: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinDecomp16BitsAreBinaryCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- Input to the circuit is just the one (combined) MLE ---
        let mut combined_signed_bin_decomp_mle = DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(self.signed_bin_decomp_mle.clone());
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut combined_signed_bin_decomp_mle)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let live_committed_input_layer: PublicInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer();

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Dataparallel/batching stuff + sanitychecks ---
        let num_subcircuit_copies = self.signed_bin_decomp_mle.len();
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;

        // --- Create the builders for (b_i)^2 - b_i ---
        let diff_builders = self.signed_bin_decomp_mle.iter_mut().map(|diff_signed_bin_decomp_mle| {
            diff_signed_bin_decomp_mle.set_prefix_bits(
                Some(
                    diff_signed_bin_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
                        combined_signed_bin_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
                            repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                        )
                    ).collect_vec()
                )
            );
            
            from_mle(
                diff_signed_bin_decomp_mle, 
                |diff_signed_bin_decomp_mle| {
                    let combined_bin_decomp_mle_ref = diff_signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
                    ExpressionStandard::Product(vec![combined_bin_decomp_mle_ref.clone(), combined_bin_decomp_mle_ref.clone()]) - ExpressionStandard::Mle(combined_bin_decomp_mle_ref)
                }, 
                |mle, id, prefix_bits| {
                    ZeroMleRef::new(mle.num_iterated_vars(), prefix_bits, id)
            })
        }).collect_vec();
        let combined_output_zero_mle_ref = combine_zero_mle_ref(layers.add_gkr(BatchedLayer::new(diff_builders)));

        Witness { layers, output_layers: vec![combined_output_zero_mle_ref.get_enum()], input_layers: vec![live_committed_input_layer.to_enum()] }
    }
}

impl<F: FieldExt> BinDecomp16BitsAreBinaryCircuit<F> {
    /// Creates a new instance of BinDecomp16BitsAreBinaryCircuit
    pub fn new(
        batched_diff_signed_bin_decomp_mle: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    ) -> Self {
        Self {
            signed_bin_decomp_mle: batched_diff_signed_bin_decomp_mle,
        }
    }
}

pub fn recompute_16_bit_decomp<F: FieldExt>(
    decomp_bits: &[F; 16],
) -> F {
    // skip 1 because the last bit is the signed bit
    decomp_bits.iter().rev().enumerate().fold(
        F::zero(), |acc, (bit_idx, bit)| {
        acc + *bit * F::from(2_u64.pow((16 - (bit_idx + 1)) as u32))
    })
}

pub fn generate_16_bit_decomp<F: FieldExt>(
    sample_size: usize,
    in_features: usize,
) -> (
    Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    Vec<DenseMle<F, F>>,
) {
    let mut rng = test_rng();

    let bin_decomp_16_bits: Vec<Vec<[F; 16]>> = (0..sample_size).map(
        |_| (0..in_features).map(
            |_| (0..16).map(
                |_| F::from(rng.gen_range(0..=1))
            ).collect_vec().try_into().unwrap()
        ).collect()
    ).collect();

    let mle_bin_decomp_16_bits = bin_decomp_16_bits.clone().into_iter().map(
        |sample| DenseMle::new_from_iter(
                sample.into_iter().map(
                    |in_feature| BinDecomp16Bit {
                        bits: in_feature,
                    }
                ), LayerId::Input(0), None
        )
    ).collect_vec();

    let bin_decomp_recomp: Vec<Vec<F>> = bin_decomp_16_bits.iter().map(
        |sample| sample.iter().map(
            |in_feature| recompute_16_bit_decomp(in_feature)
        ).collect()
    ).collect();

    let mle_bin_decomp_recomp = bin_decomp_recomp.into_iter().map(
        |sample| DenseMle::new_from_iter(
                sample.into_iter(),
                LayerId::Input(0),
                None,
        )
    ).collect();

    (mle_bin_decomp_16_bits, mle_bin_decomp_recomp)
}

#[test]
fn test_bin_decomp_16_bit_bits_are_binary() {

    // generate a random 16 bits, compute the number from the decomposition
    let (
        binary_decomp_mle_vec,
        _binary_recomp_mle_vec
    ) = generate_16_bit_decomp::<Fr>(4, 2);

    let mut circuit = BinDecomp16BitsAreBinaryCircuit::new(
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