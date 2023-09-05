use ark_std::{log2, test_rng, One, UniformRand, Zero};
use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use itertools::Itertools;
use rand::Rng;
use remainder_ligero::ligero_commit::remainder_ligero_commit_prove;
use serde_json::{from_reader, to_writer};
use std::{cmp::max, fs, iter::repeat_with, path::Path, time::Instant};
use tracing::Level;

use crate::{
    expression::ExpressionStandard,
    layer::{from_mle, layer_enum::LayerEnum, LayerBuilder, LayerId, empty_layer::EmptyLayer, batched::{combine_mles_refs, BatchedLayer, combine_zero_mle_ref}},
    mle::{
        dense::{DenseMle, Tuple2},
        mle_enum::MleEnum,
        zero::ZeroMleRef,
        Mle, MleRef, MleIndex,
    },
    prover::input_layer::{enum_input_layer::CommitmentEnum, MleInputLayer},
    utils::get_random_mle,
    zkdt::{
        structs::{BinDecomp16Bit, DecisionNode, InputAttribute, LeafNode, combine_mle_refs},
        zkdt_layer::{
            AttributeConsistencyBuilder, BitExponentiationBuilder, ConcatBuilder,
            DecisionPackingBuilder, EqualityCheck, InputPackingBuilder, LeafPackingBuilder,
            ProductBuilder, RMinusXBuilder, SplitProductBuilder, SquaringBuilder, ZeroBuilder,
        },
    },
};
use remainder_shared_types::{
    transcript::{poseidon_transcript::PoseidonTranscript, Transcript},
    FieldExt,
};

use super::{
    combine_layers::combine_layers,
    input_layer::{
        combine_input_layers::InputLayerBuilder, enum_input_layer::InputLayerEnum,
        ligero_input_layer::LigeroInputLayer, public_input_layer::PublicInputLayer,
        random_input_layer::RandomInputLayer, InputLayer,
    },
    GKRCircuit, GKRError, Layers, Witness, test_helper_circuits::{EmptyLayerBuilder, EmptyLayerSubBuilder, EmptyLayerAddBuilder},
};

fn test_circuit<F: FieldExt, C: GKRCircuit<F>>(mut circuit: C, path: Option<&Path>) {
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

/// This circuit is a 4 --> 2 circuit, such that
/// [x_1, x_2, x_3, x_4, (y_1, y_2)] --> [x_1 * x_3, x_2 * x_4] --> [x_1 * x_3 - y_1, x_2 * x_4 - y_2]
/// Note that we also have the difference thingy (of size 2)
struct SimpleCircuit<F: FieldExt> {
    mle: DenseMle<F, Tuple2<F>>,
    size: usize,
}
impl<F: FieldExt> GKRCircuit<F> for SimpleCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle)];
        let mut input_layer =
            InputLayerBuilder::new(input_mles, Some(vec![self.size]), LayerId::Input(0));
        let mle_clone = &self.mle;

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        // --- Create a SimpleLayer from the first `mle` within the circuit ---
        let mult_builder = from_mle(
            mle_clone,
            // --- The expression is a simple product between the first and second halves ---
            |mle| ExpressionStandard::products(vec![mle.first(), mle.second()]),
            // --- The witness generation simply zips the two halves and multiplies them ---
            |mle, layer_id, prefix_bits| {
                DenseMle::new_from_iter(
                    mle.into_iter()
                        .map(|Tuple2((first, second))| first * second),
                    layer_id,
                    prefix_bits,
                )
            },
        );

        // --- Stacks the two aforementioned layers together into a single layer ---
        // --- Then adds them to the overall circuit ---
        let first_layer_output = layers.add_gkr(mult_builder);

        // --- Ahh. So we're doing the thing where we add the "real" circuit output as a circuit input, ---
        // --- then check if the difference between the two is zero ---
        let mut output_input =
            DenseMle::new_from_iter(first_layer_output.into_iter(), LayerId::Input(0), None);

        // --- Index the input-output layer ONLY for the input ---
        input_layer.add_extra_mle(Box::new(&mut output_input));

        // --- Subtract the computed circuit output from the advice circuit output ---
        let output_diff_builder = from_mle(
            (first_layer_output, output_input.clone()),
            |(mle1, mle2)| mle1.mle_ref().expression() - mle2.mle_ref().expression(),
            |(mle1, mle2), layer_id, prefix_bits| {
                let num_vars = max(mle1.num_iterated_vars(), mle2.num_iterated_vars());
                ZeroMleRef::new(num_vars, prefix_bits, layer_id)
            },
        );

        // --- Add this final layer to the circuit ---
        let circuit_output = layers.add_gkr(output_diff_builder);

        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer.to_input_layer();

        let input_layers = vec![input_layer.to_enum()];

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers,
        }
    }
}

#[test]
fn test_gkr_simple_circuit() {
    let mut rng = test_rng();
    let size = 5;

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        (0..1 << 5).map(|_| (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())).into()),
        LayerId::Input(0),
        None,
    );
    // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
    //     (0..size).map(|idx| (Fr::from(idx + 2), Fr::from(idx + 2)).into()),
    //     LayerId::Input(0),
    //     None,
    // );

    let mut circuit: SimpleCircuit<Fr> = SimpleCircuit { mle, size };
    test_circuit(circuit, None);
}

/// Circuit which just subtracts its two halves! No input-output layer needed.
struct SimplestCircuit<F: FieldExt> {
    mle: DenseMle<F, Tuple2<F>>,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle)];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let mle_clone = self.mle.clone();

        // --- Create Layers to be added to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Create a SimpleLayer from the first `mle` within the circuit ---
        let diff_builder = from_mle(
            mle_clone,
            // --- The expression is a simple diff between the first and second halves ---
            |mle| {
                let first_half = Box::new(ExpressionStandard::Mle(mle.first()));
                let second_half = Box::new(ExpressionStandard::Mle(mle.second()));
                let negated_second_half = Box::new(ExpressionStandard::Negated(second_half));
                ExpressionStandard::Sum(first_half, negated_second_half)
            },
            // --- The witness generation simply zips the two halves and subtracts them ---
            |mle, layer_id, prefix_bits| {
                // DenseMle::new_from_iter(
                //     mle.into_iter()
                //         .map(|Tuple2((first, second))| first - second),
                //     layer_id,
                //     prefix_bits,
                // )
                // --- The output SHOULD be all zeros ---
                let num_vars = max(mle.first().num_vars(), mle.second().num_vars());
                ZeroMleRef::new(num_vars, prefix_bits, layer_id)
            },
        );

        // --- Stacks the two aforementioned layers together into a single layer ---
        // --- Then adds them to the overall circuit ---
        let first_layer_output = layers.add_gkr(diff_builder);

        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer.to_input_layer();

        // (layers, vec![first_layer_output.get_enum()], input_layer)
        Witness {
            layers,
            output_layers: vec![first_layer_output.get_enum()],
            input_layers: vec![input_layer.to_enum()],
        }
    }
}

#[test]
fn test_gkr_simplest_circuit() {
    let mut rng = test_rng();
    let size = 1 << 4;

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        (0..size).map(|_| {
            let num = Fr::from(rng.gen::<u64>());
            //let second_num = Fr::from(rng.gen::<u64>());
            (num, num).into()
        }),
        LayerId::Input(0),
        None,
    );
    // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
    //     (0..size).map(|idx| (Fr::from(idx + 1), Fr::from(idx + 1)).into()),
    //     LayerId::Input(0),
    //     None,
    // );

    let mut circuit: SimplestCircuit<Fr> = SimplestCircuit { mle };

    test_circuit(circuit, None);

    // panic!();
}

/// Circuit which just subtracts its two halves! No input-output layer needed.
struct SimplestBatchedCircuit<F: FieldExt> {
    batched_mle: Vec<DenseMle<F, Tuple2<F>>>,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestBatchedCircuit<F> {

    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        let all_first_seconds = self.batched_mle.clone().into_iter().map(|mle| {
            combine_mles_refs(vec![mle.first(), mle.second()], 1)
        }).collect_vec();

        let mut combined_all = combine_mle_refs(all_first_seconds);

        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut combined_all)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));

        // --- Create Layers to be added to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Create a SimpleLayer from the first `mle` within the circuit ---
        let diff_builders = self.batched_mle.clone().into_iter().map(
            |mle| {
                let diff_builder = from_mle(
                    mle,
                    // --- The expression is a simple diff between the first and second halves ---
                    |mle| {
                        let first_half = ExpressionStandard::Mle(mle.first());
                        let second_half = ExpressionStandard::Mle(mle.second());
                        first_half - second_half
                    },
                    // --- The witness generation simply zips the two halves and subtracts them ---
                    |mle, layer_id, prefix_bits| {
                        // let hi = DenseMle::new_from_iter(
                        //     mle.into_iter()
                        //         .map(|Tuple2((first, second))| first - second),
                        //     layer_id,
                        //     prefix_bits.clone(),
                        // );
                        // dbg!(&hi.mle_ref().bookkeeping_table);
                        // --- The output SHOULD be all zeros ---
                        let num_vars = max(mle.first().num_vars(), mle.second().num_vars());
                        ZeroMleRef::new(num_vars, prefix_bits, layer_id)
                    },
                );
                diff_builder
            }
        ).collect_vec();
        

        let batched_builder = BatchedLayer::new(diff_builders);

        let batched_result = layers.add_gkr(batched_builder);


        let batched_zero = combine_zero_mle_ref(batched_result);

        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_layer: PublicInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer();

        // (layers, vec![first_layer_output.get_enum()], input_layer)
        Witness { layers, output_layers: vec![batched_zero.get_enum()], input_layers: vec![input_layer.to_enum()] }
    }
}

#[test]
fn test_gkr_simplest_batched_circuit() {
    let mut rng = test_rng();
    let size = 1 << 3;

    let batch_size = 1 << 2;
    // --- This should be 2^2 ---
    let batched_mle: Vec<DenseMle<Fr, Tuple2<Fr>>> = (0..batch_size).map(|idx1| {
        DenseMle::new_from_iter(
        (0..size).map(|idx| {
            let num = Fr::from(rng.gen::<u64>());
            //let second_num = Fr::from(rng.gen::<u64>());
            // let num = Fr::from(idx + idx1);
            (num, num).into()
        }),
        LayerId::Input(0),
        None)
    }).collect_vec();
    // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
    //     (0..size).map(|idx| (Fr::from(idx + 1), Fr::from(idx + 1)).into()),
    //     LayerId::Input(0),
    //     None,
    // );

    let circuit: SimplestBatchedCircuit<Fr> = SimplestBatchedCircuit { batched_mle };

    test_circuit(circuit, None);

    // panic!();
}

///This circuit checks how RandomLayer works by multiplying the MLE by a constant, taking in that result as advice in a publiclayer and doing an equality check on the result of the mult and the advice
struct RandomCircuit<F: FieldExt> {
    mle: DenseMle<F, F>,
}

impl<F: FieldExt> GKRCircuit<F> for RandomCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }

    fn synthesize_and_commit(
        &mut self,
        transcript: &mut Self::Transcript,
    ) -> Result<(Witness<F, Self::Transcript>, Vec<CommitmentEnum<F>>), GKRError> {
        let mut input =
            InputLayerBuilder::new(vec![Box::new(&mut self.mle)], None, LayerId::Input(0))
                .to_input_layer::<LigeroInputLayer<F, _>>()
                .to_enum();

        let input_commit = input
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;
        InputLayerEnum::append_commitment_to_transcript(&input_commit, transcript).unwrap();

        let random = RandomInputLayer::new(transcript, 1, LayerId::Input(1));
        let random_mle = random.get_mle();
        let mut random = random.to_enum();
        let random_commit = random
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;

        let mut layers = Layers::new();

        let layer_1 = from_mle(
            (self.mle.clone(), random_mle),
            |(mle, random)| ExpressionStandard::products(vec![mle.mle_ref(), random.mle_ref()]),
            |(mle, random), layer_id, prefix_bits| {
                DenseMle::new_from_iter(
                    mle.into_iter()
                        .zip(random.into_iter().cycle())
                        .map(|(item, random)| item * random),
                    layer_id,
                    prefix_bits,
                )
            },
        );

        let output = layers.add_gkr(layer_1);

        let mut output_input = output.clone();
        output_input.layer_id = LayerId::Input(2);
        let mut input_layer_2 =
            InputLayerBuilder::new(vec![Box::new(&mut output_input)], None, LayerId::Input(2))
                .to_input_layer::<PublicInputLayer<F, _>>()
                .to_enum();
        let input_layer_2_commit = input_layer_2
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;
        InputLayerEnum::append_commitment_to_transcript(&input_layer_2_commit, transcript).unwrap();

        let layer_2 = EqualityCheck::new(output, output_input);
        let output = layers.add_gkr(layer_2);

        Ok((
            Witness {
                layers,
                output_layers: vec![output.get_enum()],
                input_layers: vec![input, random, input_layer_2],
            },
            vec![input_commit, random_commit, input_layer_2_commit],
        ))
    }
}

#[test]
fn test_random_layer_circuit() {
    // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
    // tracing::subscriber::set_global_default(subscriber)
    //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

    let num_vars = 5;
    let mle = get_random_mle::<Fr>(num_vars);

    let mut circuit = RandomCircuit { mle };

    test_circuit(circuit, Some(Path::new("./random_proof.json")));
}

/// This circuit is a 4k --> k circuit, such that
/// [x_1, x_2, x_3, x_4] --> [x_1 * x_3, x_2 + x_4] --> [(x_1 * x_3) - (x_2 + x_4)]
struct TestCircuit<F: FieldExt> {
    mle: DenseMle<F, Tuple2<F>>,
    mle_2: DenseMle<F, Tuple2<F>>,
    size: usize,
}

impl<F: FieldExt> GKRCircuit<F> for TestCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The input layer should just be the concatenation of `mle`, `mle_2`, and `output_input` ---
        // let mut self_mle_clone = self.mle.clone();
        // let mut self_mle_2_clone = self.mle_2.clone();
        let mut input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.mle), Box::new(&mut self.mle_2)];
        let mut input_layer =
            InputLayerBuilder::new(input_mles, Some(vec![self.size]), LayerId::Input(0));
        let mle_clone = self.mle.clone();
        let mle_2_clone = self.mle_2.clone();

        // --- Create Layers to be added to ---

        // --- Create Layers to be added to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Create a SimpleLayer from the first `mle` within the circuit ---
        let builder = from_mle(
            mle_clone,
            // --- The expression is a simple product between the first and second halves ---
            |mle| ExpressionStandard::products(vec![mle.first(), mle.second()]),
            // --- The witness generation simply zips the two halves and multiplies them ---
            |mle, layer_id, prefix_bits| {
                DenseMle::new_from_iter(
                    mle.into_iter()
                        .map(|Tuple2((first, second))| first * second),
                    layer_id,
                    prefix_bits,
                )
            },
        );

        // --- Similarly here, but with addition between the two halves ---
        // --- Note that EACH of `mle` and `mle_2` are parts of the input layer ---
        let builder2 = from_mle(
            mle_2_clone,
            |mle| mle.first().expression() + mle.second().expression(),
            |mle, layer_id, prefix_bits| {
                DenseMle::new_from_iter(
                    mle.into_iter()
                        .map(|Tuple2((first, second))| first + second),
                    layer_id,
                    prefix_bits,
                )
            },
        );

        // --- Stacks the two aforementioned layers together into a single layer ---
        // --- Then adds them to the overall circuit ---
        let builder3 = builder.concat(builder2);
        let output = layers.add_gkr(builder3);

        // --- Creates a single layer which takes [x_1, ..., x_n, y_1, ..., y_n] and returns [x_1 - y_1, ..., x_n - y_n] ---
        let builder4 = from_mle(
            output,
            |(mle1, mle2)| mle1.mle_ref().expression() - mle2.mle_ref().expression(),
            |(mle1, mle2), layer_id, prefix_bits| {
                DenseMle::new_from_iter(
                    mle1.clone()
                        .into_iter()
                        .zip(mle2.clone().into_iter())
                        .map(|(first, second)| first - second),
                    layer_id,
                    prefix_bits,
                )
            },
        );

        // --- Appends this to the circuit ---
        let computed_output = layers.add_gkr(builder4);

        // --- Ahh. So we're doing the thing where we add the "real" circuit output as a circuit input, ---
        // --- then check if the difference between the two is zero ---
        let mut output_input =
            DenseMle::new_from_iter(computed_output.into_iter(), LayerId::Input(0), None);
        input_layer.add_extra_mle(Box::new(&mut output_input));

        // --- Subtract the computed circuit output from the advice circuit output ---
        let builder5 = from_mle(
            (computed_output, output_input.clone().clone()),
            |(mle1, mle2)| mle1.mle_ref().expression() - mle2.mle_ref().expression(),
            |(mle1, mle2), layer_id, prefix_bits| {
                let num_vars = max(mle1.num_iterated_vars(), mle2.num_iterated_vars());
                ZeroMleRef::new(num_vars, prefix_bits, layer_id)
            },
        );

        // --- Add this final layer to the circuit ---
        let circuit_circuit_output = layers.add_gkr(builder5);

        // --- The input layer should just be the concatenation of `mle`, `mle_2`, and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.mle), Box::new(&mut self.mle_2)];

        let input_layer: PublicInputLayer<F, Self::Transcript> = input_layer.to_input_layer();

        // (layers, vec![circuit_circuit_output.get_enum()], input_layer)
        Witness {
            layers,
            output_layers: vec![circuit_circuit_output.get_enum()],
            input_layers: vec![input_layer.to_enum()],
        }
    }
}

#[test]
fn test_gkr() {
    let mut rng = test_rng();
    let size = 4;
    let size_expanded = 1 << size;
    // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
    // tracing::subscriber::set_global_default(subscriber)
    //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        (0..size_expanded).map(|_| (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())).into()),
        LayerId::Input(0),
        None,
    );
    // --- This should be 2^2 ---
    let mle_2: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        (0..size_expanded).map(|_| (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())).into()),
        LayerId::Input(0),
        None,
    );

    let mut circuit: TestCircuit<Fr> = TestCircuit { mle, mle_2, size };

    test_circuit(circuit, Some(Path::new("./gkr_proof.json")));
}

/// Circuit which just subtracts its two halves with gate mle
struct SimplestGateCircuit<F: FieldExt> {
    mle: DenseMle<F, F>,
    negmle: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestGateCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let mut input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle), Box::new(&mut self.negmle)];
        let mut input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0)).to_input_layer::<PublicInputLayer<F, _>>().to_enum();
        let mle_clone = self.mle.clone();

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let num_vars = self.mle.mle_ref().num_vars();

        (0..num_vars).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let first_layer_output =
            layers.add_add_gate(nonzero_gates, self.mle.mle_ref(), self.negmle.mle_ref(), 0);

        let output_layer_builder = ZeroBuilder::new(first_layer_output);

        let output_layer_mle = layers.add_gkr(output_layer_builder);

        // (layers, vec![first_layer_output.mle_ref().get_enum()], input_layer)
        Witness {layers, output_layers: vec![output_layer_mle.get_enum()], input_layers: vec![input_layer]}
    }
}

/// Circuit which just subtracts its two halves with batched gate mle
struct SimplestBatchedGateCircuit<F: FieldExt> {
    mle: DenseMle<F, F>,
    negmle: DenseMle<F, F>,
    batch_bits: usize,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestBatchedGateCircuit<F> {

    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle), Box::new(&mut self.negmle)];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0)).to_input_layer::<PublicInputLayer<F, _>>().to_enum();

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let num_vars = self.mle.mle_ref().num_vars();

        (0..num_vars).for_each(
            |idx| {
                nonzero_gates.push((idx, idx, idx));
            }
        );

        let first_layer_output = layers.add_add_gate(nonzero_gates, self.mle.mle_ref(), self.negmle.mle_ref(), 0);

        let output_layer_builder = ZeroBuilder::new(first_layer_output);

        let output_layer_mle = layers.add_gkr(output_layer_builder);

        // (layers, vec![first_layer_output.mle_ref().get_enum()], input_layer)
        Witness {layers, output_layers: vec![output_layer_mle.get_enum()], input_layers: vec![input_layer]}
    }
}

#[test]
fn test_gkr_gate_simplest_circuit() {
    let mut rng = test_rng();
    let size = 1 << 4;

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| {
            let num = Fr::from(rng.gen::<u64>());
            num
        }),
        LayerId::Input(0),
        None,
    );

    let negmle = DenseMle::new_from_iter(
        mle.mle_ref()
            .bookkeeping_table
            .into_iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        None,
    );
    // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
    //     (0..size).map(|idx| (Fr::from(idx + 1), Fr::from(idx + 1)).into()),
    //     LayerId::Input,
    //     None,
    // );

    let mut circuit: SimplestGateCircuit<Fr> = SimplestGateCircuit { mle, negmle };

    test_circuit(circuit, Some(Path::new("./gate_proof.json")));

    // panic!();
}

#[test]
fn test_gkr_gate_batched_simplest_circuit() {
    let mut rng = test_rng();
    let size = 1 << 4;

    // --- This should be 2^4 ---
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| {
            let num = Fr::from(rng.gen::<u64>());
            num
        }),
        LayerId::Input(0),
        // this is the batched bits
        Some(vec![MleIndex::Iterated; 2]),
    );

    let negmle = DenseMle::new_from_iter(
        mle.mle_ref().bookkeeping_table.into_iter().map(
            |elem|
            -elem
        ), 
        LayerId::Input(0),
        // this is the batched bits
        Some(vec![MleIndex::Iterated; 2]),
    );
    // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
    //     (0..size).map(|idx| (Fr::from(idx + 1), Fr::from(idx + 1)).into()),
    //     LayerId::Input,
    //     None,
    // );

    let circuit: SimplestBatchedGateCircuit<Fr> = SimplestBatchedGateCircuit { mle, negmle, batch_bits: 2 };

    test_circuit(circuit, Some(Path::new("./gate_batch_proof.json")));

    // panic!();
}

/// Circuit which subtracts its two halves, except for the part where one half is
/// comprised of a pre-committed Ligero input layer and the other half is comprised
/// of a Ligero input layer which is committed to on the spot.
/// 
/// The circuit itself produces independent claims on its two input MLEs, and is basically
/// two indpendent circuits via the fact that it basically subtracts each input MLE
/// from itself and calls that the output layer. In particular, this allows us to test
/// whether Halo2 generates the same VK given that we have the same pre-committed Ligero layer
/// but a DIFFERENT live-committed Ligero layer
struct SimplePrecommitCircuit<F: FieldExt> {
    mle: DenseMle<F, F>,
    mle2: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for SimplePrecommitCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The precommitted input layer MLE is just the first MLE ---
        let precommitted_input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle)];
        let precommitted_input_layer_builder =
            InputLayerBuilder::new(precommitted_input_mles, None, LayerId::Input(0));

        // --- The non-precommitted input layer MLE is just the second ---
        let live_committed_input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle2)];
        let live_committed_input_layer_builder =
            InputLayerBuilder::new(live_committed_input_mles, None, LayerId::Input(1));

        let mle_clone = self.mle.clone();
        let mle2_clone = self.mle2.clone();

        // --- Create Layers to be added to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Create a SimpleLayer from the first `mle` within the circuit ---
        let diff_builder = from_mle(
            mle_clone.clone(),
            // --- The expression is a simple diff between the first and second halves ---
            |_mle| {
                let first_half = ExpressionStandard::Mle(_mle.mle_ref());
                let second_half = ExpressionStandard::Mle(_mle.mle_ref());
                first_half - second_half
            },
            // --- The output SHOULD be all zeros ---
            |_mle, layer_id, prefix_bits| {
                ZeroMleRef::new(mle_clone.mle_ref().num_vars(), prefix_bits, layer_id)
            },
        );

        // --- Similarly as the above, but with the circuit's second MLE ---
        let diff_builder_2 = from_mle(
            mle2_clone.clone(),
            // --- The expression is a simple diff between the first and second halves ---
            |_mle| {
                let first_half = ExpressionStandard::Mle(_mle.mle_ref());
                let second_half = ExpressionStandard::Mle(_mle.mle_ref());
                first_half - second_half
            },
            // --- The output SHOULD be all zeros ---
            |_mle, layer_id, prefix_bits| {
                ZeroMleRef::new(mle2_clone.mle_ref().num_vars(), prefix_bits, layer_id)
            },
        );

        // --- Stacks the two aforementioned layers together into a single layer ---
        // --- Then adds them to the overall circuit ---
        let first_layer_output_1 = layers.add_gkr(diff_builder);
        let first_layer_output_2 = layers.add_gkr(diff_builder_2);

        // --- We should have two input layers: a single pre-committed and a single regular Ligero layer ---
        let rho_inv = 4;
        let (_, ligero_comm, ligero_root, ligero_aux) =
            remainder_ligero_commit_prove(&self.mle.mle, rho_inv);
        let precommitted_input_layer: LigeroInputLayer<F, Self::Transcript> =
            precommitted_input_layer_builder.to_input_layer_with_precommit(
                ligero_comm,
                ligero_aux,
                ligero_root,
            );
        let live_committed_input_layer: LigeroInputLayer<F, Self::Transcript> =
            live_committed_input_layer_builder.to_input_layer();

        Witness {
            layers,
            output_layers: vec![first_layer_output_1.get_enum(), first_layer_output_2.get_enum()],
            input_layers: vec![
                precommitted_input_layer.to_enum(),
                live_committed_input_layer.to_enum(),
            ],
        }
    }
}

#[test]
fn test_gkr_circuit_with_precommit() {
    let mut rng = test_rng();
    let size = 1 << 5;

    // --- MLE contents ---
    let items = repeat_with(|| Fr::from(rng.gen::<u64>()))
        .take(size)
        .collect_vec();
    let items2 = repeat_with(|| Fr::from(rng.gen::<u64>()))
        .take(size)
        .collect_vec();

    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(items, LayerId::Input(0), None);
    let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(items2, LayerId::Input(1), None);

    let mut circuit: SimplePrecommitCircuit<Fr> = SimplePrecommitCircuit { mle, mle2 };

    test_circuit(circuit, Some(Path::new("./gkr_proof_with_precommit.json")));
}

/// Circuit which has an empty layer as an intermediate layer with multiple claims going both
/// to and from it, to thoroughly test the expected behavior of the `EmptyLayer`.
/// 
/// Note that all three MLEs have size 2! The structure of the circuit is as follows:
/// * The two empty layer src MLEs' elements are each multiplied together and then added between
/// the two to yield a single empty layer.
/// * Next, that empty layer's MLE is subtracted from the first MLE
/// * Finally, that empty layer's MLE is added to the second MLE
struct EmptyLayerTestCircuit<F: FieldExt> {
    mle: DenseMle<F, F>,
    mle2: DenseMle<F, F>,
    empty_layer_src_mle: DenseMle<F, F>,
    other_empty_layer_src_mle: DenseMle<F, F>
}
impl<F: FieldExt> GKRCircuit<F> for EmptyLayerTestCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- We're not testing commitments here; just use PublicInputLayer ---
        let input_layer =
            InputLayerBuilder::new(vec![
                Box::new(&mut self.mle),
                Box::new(&mut self.mle2),
                Box::new(&mut self.empty_layer_src_mle),
                Box::new(&mut self.other_empty_layer_src_mle)
            ], None, LayerId::Input(0)).to_input_layer::<PublicInputLayer<F, _>>();

        // --- Create Layers to be added to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Creates empty layer and adds to circuit ---
        let empty_layer_builder = EmptyLayerBuilder::new(self.empty_layer_src_mle.clone(), self.other_empty_layer_src_mle.clone());
        let empty_layer_result = layers.add::<_, EmptyLayer<F, Self::Transcript>>(empty_layer_builder);

        // --- Subtracts from `self.mle` ---
        let sub_builder = EmptyLayerSubBuilder::new(empty_layer_result.clone(), self.mle.clone());
        let sub_result = layers.add_gkr(sub_builder);

        // --- Adds to `self.mle2` ---
        let add_builder = EmptyLayerAddBuilder::new(empty_layer_result, self.mle2.clone());
        let add_result = layers.add_gkr(add_builder);

        Witness {
            layers,
            output_layers: vec![sub_result.get_enum(), add_result.get_enum()],
            input_layers: vec![
                input_layer.to_enum(),
            ],
        }
    }
}

#[test]
fn test_empty_layer_circuit() {

    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(vec![Fr::from(14), Fr::from(14)], LayerId::Input(0), None);
    let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(vec![Fr::from(14).neg(), Fr::from(14).neg()], LayerId::Input(0), None);
    let empty_layer_src_mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(vec![Fr::from(1), Fr::from(2)], LayerId::Input(0), None);
    let other_empty_layer_src_mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(vec![Fr::from(3), Fr::from(4)], LayerId::Input(0), None);

    let circuit: EmptyLayerTestCircuit<Fr> = EmptyLayerTestCircuit { mle, mle2, empty_layer_src_mle, other_empty_layer_src_mle };

    test_circuit(circuit, None);
}