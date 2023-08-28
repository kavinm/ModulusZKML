use ark_std::log2;
use itertools::{Itertools, repeat_n};

use crate::{prover::{GKRCircuit, Layers, input_layer::InputLayer, Witness}, mle::{dense::DenseMle, MleRef, beta::BetaTable, Mle, MleIndex, mle_enum::MleEnum}, layer::{LayerBuilder, empty_layer::EmptyLayer, batched::{BatchedLayer, combine_zero_mle_ref}, LayerId}, sumcheck::{compute_sumcheck_message, Evals, get_round_degree}};
use remainder_shared_types::{FieldExt, transcript::{Transcript, poseidon_transcript::PoseidonTranscript}};

use super::{zkdt_layer::{InputPackingBuilder, SplitProductBuilder, EqualityCheck, AttributeConsistencyBuilder, DecisionPackingBuilder, LeafPackingBuilder, ConcatBuilder, RMinusXBuilder, BitExponentiationBuilder, SquaringBuilder, ProductBuilder, PrevNodeLeftBuilderDecision, PrevNodeRightBuilderDecision, CurrNodeBuilderDecision, CurrNodeBuilderLeaf, SignBit, OneMinusSignBit, SignBitProductBuilder}, structs::{InputAttribute, DecisionNode, LeafNode, BinDecomp16Bit}};

pub struct PermutationCircuit<F: FieldExt> {
    pub dummy_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,               // batched
    pub dummy_permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,      // batched
    pub r: F,
    pub r_packing: F,
    pub input_len: usize,
    pub num_inputs: usize
}

impl<F: FieldExt> GKRCircuit<F> for PermutationCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        // layer 0: packing
        // let input_packing_builder: InputPackingBuilder<F> = InputPackingBuilder::new(
        //     self.dummy_input_data_mle_vec[0].clone(),
        //     self.r,
        //     self.r_packing);

        // let input_permuted_packing_builder: InputPackingBuilder<F> = InputPackingBuilder::new(
        //     self.dummy_permuted_input_data_mle_vec[0].clone(),
        //     self.r,
        //     self.r_packing);

        let batch_bits = log2(self.dummy_input_data_mle_vec.len()) as usize;
    
    
        let input_packing_builder = BatchedLayer::new(self.dummy_input_data_mle_vec.iter().map(|input_data_mle| {
            let mut input_data_mle = input_data_mle.clone();
            input_data_mle.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));
            InputPackingBuilder::new(
                input_data_mle,
                self.r,
                self.r_packing
            )
        }).collect_vec());

        let input_permuted_packing_builder = BatchedLayer::new(self.dummy_permuted_input_data_mle_vec.iter().map(|input_data_mle| {
            let mut input_data_mle = input_data_mle.clone();
            input_data_mle.add_prefix_bits(Some(repeat_n(MleIndex::Iterated, batch_bits).collect_vec()));
            InputPackingBuilder::new(
                input_data_mle,
                self.r,
                self.r_packing
            )
        }).collect_vec());

        let packing_builders = input_packing_builder.concat(input_permuted_packing_builder);
        // let mut expression: crate::expression::ExpressionStandard<F> = packing_builders.build_expression();
        // let num_vars = expression.index_mle_indices(0);
        // let degree = get_round_degree(&expression, 0);    
        // let (_, next) = packing_builders.next_layer(LayerId::Layer(0), None);

        // let eval = {
        //     let mut beta = BetaTable::new((vec![F::zero(), F::one(), F::zero(), F::zero(), F::zero(), F::zero(), F::zero(), F::zero(), F::zero()], F::zero())).unwrap();
        //     beta.table.index_mle_indices(0);
        //     let eval = compute_sumcheck_message(&expression, 0, degree, &beta).unwrap();
        //     let Evals(evals) = eval;
        //     evals[0] + evals[1]
        // };

        // let next = next.mle_ref();
        // let val = next.bookkeeping_table()[1];
        // dbg!(eval, val);

        // assert_eq!(eval, val);

        let (mut input_packed, mut input_permuted_packed) = layers.add_gkr(packing_builders);

        for _ in 0..log2(self.input_len) {
            let prod_builder = BatchedLayer::new(input_packed.into_iter().map(|input_packed| SplitProductBuilder::new(
                input_packed
            )).collect());
            let prod_permuted_builder = BatchedLayer::new(input_permuted_packed.into_iter().map(|input_permuted_packed| SplitProductBuilder::new(
                input_permuted_packed
            )).collect());
            let split_product_builders = prod_builder.concat(prod_permuted_builder);
            (input_packed, input_permuted_packed) = layers.add_gkr(split_product_builders);
        }

        let difference_builder = EqualityCheck::new_batched(
            input_packed,
            input_permuted_packed,
        );

        let difference_mle = layers.add_gkr(difference_builder);

        let difference_mle = combine_zero_mle_ref(difference_mle);

        todo!()
        // (layers, vec![difference_mle.get_enum()], todo!())
    }
}

pub struct PathCheckCircuit<F: FieldExt> {
    pub decision_node_paths_mle: DenseMle<F, DecisionNode<F>>, 
    pub leaf_node_paths_mle: DenseMle<F, LeafNode<F>>,
    pub bin_decomp_diff_mle: DenseMle<F, BinDecomp16Bit<F>>,
    pub num_copy: usize,
}

impl<F: FieldExt> GKRCircuit<F> for PathCheckCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>, InputLayer<F>) {
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        let prev_node_left_builder = PrevNodeLeftBuilderDecision::new(
            self.decision_node_paths_mle.clone());

        let prev_node_right_builder = PrevNodeRightBuilderDecision::new(
                self.decision_node_paths_mle.clone());

        let curr_node_decision_builder = CurrNodeBuilderDecision::new(
            self.decision_node_paths_mle.clone());

        let curr_node_leaf_builder = CurrNodeBuilderLeaf::new(
            self.leaf_node_paths_mle.clone());

        let decision_leaf_diff = self.decision_node_paths_mle.num_iterated_vars() - self.leaf_node_paths_mle.num_iterated_vars();

        let curr_node_decision_leaf = curr_node_decision_builder.concat_with_padding(curr_node_leaf_builder, Padding::Right(decision_leaf_diff));
        let (curr_decision, curr_leaf) = layers.add_gkr(curr_node_decision_leaf);

        let curr_node_decision_leaf_builder = ConcatBuilder::new(curr_decision, curr_leaf);
        
        let prev_node_left_mle_ref = layers.add_gkr(prev_node_left_builder).mle_ref();
        let prev_node_right_mle_ref = layers.add_gkr(prev_node_right_builder).mle_ref();
        let curr_node_decision_leaf_mle_ref = layers.add_gkr(curr_node_decision_leaf_builder).mle_ref();

        let nonzero_gates = create_wiring_from_num_bits(curr_node_decision_leaf_mle_ref.num_vars() - self.num_copy);

        // let res_negative = layers.add_add_gate_batched(self.num_copy, nonzero_gates.clone(), curr_node_decision_leaf_mle_ref.clone(), prev_node_left_mle_ref);
        // let res_positive = layers.add_add_gate_batched(self.num_copy, nonzero_gates, curr_node_decision_leaf_mle_ref, prev_node_right_mle_ref);

        let res_negative = layers.add_add_gate(nonzero_gates.clone(), curr_node_decision_leaf_mle_ref.clone(), prev_node_left_mle_ref, self.num_copy);
        let res_positive = layers.add_add_gate(nonzero_gates, curr_node_decision_leaf_mle_ref, prev_node_right_mle_ref, self.num_copy);

        let neg_sign_bit_builder = SignBit::new(self.bin_decomp_diff_mle.clone());
        let neg_sign_bits = layers.add_gkr(neg_sign_bit_builder);

        let pos_sign_bit_builder = OneMinusSignBit::new(self.bin_decomp_diff_mle.clone());
        let pos_sign_bits = layers.add_gkr(pos_sign_bit_builder);

        let sign_bit_sum_builder = SignBitProductBuilder::new(pos_sign_bits, neg_sign_bits, res_positive, res_negative);
        let final_res = layers.add_gkr(sign_bit_sum_builder);

        let mut input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.decision_node_paths_mle), Box::new(&mut self.leaf_node_paths_mle), Box::new(&mut self.bin_decomp_diff_mle)];
        
        let mut input_layer = InputLayer::<F>::new_from_mles(&mut input_mles, None);
        input_layer.combine_input_mles(&input_mles, None);

        (layers, vec![Box::new(final_res)], input_layer)
    }
}

fn create_wiring_from_num_bits(num_bits: usize) -> Vec<(usize, usize, usize)> {
    (0..(num_bits-1)).into_iter().map(
        |idx| (idx, idx + 1, idx)
    ).collect_vec()
}

struct AttributeConsistencyCircuit<F: FieldExt> {
    dummy_permuted_input_data_mle_vec: DenseMle<F, InputAttribute<F>>, // batched
    dummy_decision_node_paths_mle_vec: DenseMle<F, DecisionNode<F>>,     // batched
    tree_height: usize,
}

impl<F: FieldExt> GKRCircuit<F> for AttributeConsistencyCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        let attribute_consistency_builder = AttributeConsistencyBuilder::new(
            self.dummy_permuted_input_data_mle_vec.clone(),
            self.dummy_decision_node_paths_mle_vec.clone(),
            self.tree_height
        );

        let difference_mle = layers.add_gkr(attribute_consistency_builder);


        todo!()
        // (layers, vec![difference_mle.mle_ref().get_enum()], todo!())
    }
}

struct MultiSetCircuit<F: FieldExt> {
    dummy_decision_nodes_mle: DenseMle<F, DecisionNode<F>>,
    dummy_leaf_nodes_mle: DenseMle<F, LeafNode<F>>,
    dummy_multiplicities_bin_decomp_mle: DenseMle<F, BinDecomp16Bit<F>>,
    dummy_decision_node_paths_mle_vec: DenseMle<F, DecisionNode<F>>, // batched
    dummy_leaf_node_paths_mle_vec: DenseMle<F, LeafNode<F>>,         // batched
    r: F,
    r_packings: (F, F),
    tree_height: usize,
    num_inputs: usize,
}

impl<F: FieldExt> GKRCircuit<F> for MultiSetCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        // layer 0
        let decision_packing_builder = DecisionPackingBuilder::new(
            self.dummy_decision_nodes_mle.clone(), self.r, self.r_packings);

        let leaf_packing_builder = LeafPackingBuilder::new(
            self.dummy_leaf_nodes_mle.clone(), self.r, self.r_packings.0
        );

        let packing_builders = decision_packing_builder.concat(leaf_packing_builder);
        let (decision_packed, leaf_packed) = layers.add_gkr(packing_builders);

        // layer 1
        let decision_leaf_concat_builder = ConcatBuilder::new(
            decision_packed, leaf_packed
        );
        let x_packed = layers.add_gkr(decision_leaf_concat_builder);

        // layer 2
        let r_minus_x_builder =  RMinusXBuilder::new(
            x_packed, self.r
        );
        let mut r_minus_x = layers.add_gkr(r_minus_x_builder);

        // layer 3
        let prev_prod_builder = BitExponentiationBuilder::new(
            self.dummy_multiplicities_bin_decomp_mle.clone(),
            0,
            r_minus_x.clone()
        );
        let mut prev_prod = layers.add_gkr(prev_prod_builder);

        for i in 1..16 {

            // layer 3, or i + 2
            let r_minus_x_square_builder = SquaringBuilder::new(
                r_minus_x
            );
            let r_minus_x_square = layers.add_gkr(r_minus_x_square_builder);

            // layer 4, or i + 3
            let curr_prod_builder = BitExponentiationBuilder::new(
                self.dummy_multiplicities_bin_decomp_mle.clone(),
                i,
                r_minus_x_square.clone()
            );
            let curr_prod = layers.add_gkr(curr_prod_builder);

            // layer 5, or i + 4
            let prod_builder = ProductBuilder::new(
                curr_prod,
                prev_prod
            );
            prev_prod = layers.add_gkr(prod_builder);

            r_minus_x = r_minus_x_square;

        }

        let mut exponentiated_nodes = prev_prod;

        for _ in 0..self.tree_height {

            // layer 20, or i+20
            let prod_builder = SplitProductBuilder::new(
                exponentiated_nodes
            );
            exponentiated_nodes = layers.add_gkr(prod_builder);
        }

        // **** above is nodes exponentiated ****
        // **** below is all decision nodes on the path multiplied ****


        // layer 0: packing
        let decision_path_packing_builder = DecisionPackingBuilder::new(
            self.dummy_decision_node_paths_mle_vec.clone(),
            self.r,
            self.r_packings
        );

        let leaf_path_packing_builder = LeafPackingBuilder::new(
            self.dummy_leaf_node_paths_mle_vec.clone(),
            self.r,
            self.r_packings.0
        );

        let path_packing_builders = decision_path_packing_builder.concat(leaf_path_packing_builder);
        let (decision_path_packed, leaf_path_packed) = layers.add_gkr(path_packing_builders);

        // layer 1: concat
        let path_decision_leaf_concat_builder = ConcatBuilder::new(
            decision_path_packed, leaf_path_packed
        );
        let x_path_packed = layers.add_gkr(path_decision_leaf_concat_builder);

        // layer 2: r - x
        let r_minus_x_path_builder =  RMinusXBuilder::new(
            x_path_packed, self.r
        );
        let r_minus_x_path = layers.add_gkr(r_minus_x_path_builder);

        let mut path_product = r_minus_x_path;

        // layer remaining: product it together
        for _ in 0..log2(self.tree_height * self.num_inputs) {
            let prod_builder = SplitProductBuilder::new(
                path_product
            );
            path_product = layers.add_gkr(prod_builder);
        }

        let difference_builder = EqualityCheck::new(
            exponentiated_nodes,
            path_product
        );

        let difference = layers.add_gkr(difference_builder);

        // (layers, vec![difference.get_enum()], todo!())
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr as H2Fr;

    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use ark_std::{test_rng, UniformRand};
    use rand::Rng;

    use crate::{zkdt::zkdt_helpers::{DummyMles, generate_dummy_mles, NUM_DUMMY_INPUTS, DUMMY_INPUT_LEN, TREE_HEIGHT, generate_dummy_mles_batch, BatchedDummyMles}, prover::GKRCircuit};
    use super::{PermutationCircuit, AttributeConsistencyCircuit};
    use remainder_shared_types::transcript::{Transcript, poseidon_transcript::PoseidonTranscript};


    #[test]
    fn test_permutation_circuit() {
        let mut rng = test_rng();

        let BatchedDummyMles {
            dummy_input_data_mle,
            dummy_permuted_input_data_mle, ..
        } = generate_dummy_mles_batch();

        let mut circuit = PermutationCircuit {
            dummy_input_data_mle_vec: dummy_input_data_mle,
            dummy_permuted_input_data_mle_vec: dummy_permuted_input_data_mle,
            r: Fr::from(rng.gen::<u64>()),
            r_packing: Fr::from(rng.gen::<u64>()),
            input_len: DUMMY_INPUT_LEN * (TREE_HEIGHT - 1),
            num_inputs: 1,
        };

        let mut transcript = PoseidonTranscript::new("Permutation Circuit Prover Transcript");

        let now = Instant::now();

        let proof = circuit.prove(&mut transcript);

        println!("Proof generated!: Took {} seconds", now.elapsed().as_secs_f32());

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Permutation Circuit Verifier Transcript");
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
    fn test_attribute_consistency_circuit() {

        let DummyMles::<Fr> {
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle, ..
        } = generate_dummy_mles();

        let mut circuit = AttributeConsistencyCircuit {
            dummy_permuted_input_data_mle_vec: dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle_vec: dummy_decision_node_paths_mle,
            tree_height: TREE_HEIGHT,
        };

        let mut transcript = PoseidonTranscript::new("Attribute Consistency Circuit Prover Transcript");

        let proof = circuit.prove(&mut transcript);

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Attribute Consistency Circuit Verifier Transcript");
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