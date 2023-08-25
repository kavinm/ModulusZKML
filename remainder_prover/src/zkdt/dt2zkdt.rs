//! Conversion from a decision trees model to a circuit ready form.
//!
//! # Circuitizing tree models (c.f. [`CircuitizedTrees`])
//!
//! ```
//! use ark_bn254::Fr;
//!
//! let n_trees = 3;
//! let n_features = 4;
//! let depth = 6;
//! // start with some random (probably not perfect) trees with f64 leaf values
//! let trees_model = generate_trees_model(n_trees, depth, n_features, 0.5);
//! // pad the trees, assign ids and quantize the leaf values
//! let pqtrees: PaddedQuantizedTrees = (&trees_model).into();
//! // circuitize the trees (converts to DecisionNode<F>, LeafNode<F>)
//! let ctrees: CircuitizedTrees<Fr> = (&pqtrees).into();
//! ```
//!
//! # Circuitizing samples (c.f. [`CircuitizedSamples`])
//!
//! Continuing the above example:
//!
//! ```
//! // generate some samples to play with
//! let n_samples = 10;
//! let samples = generate_samples(n_samples, n_features);
//! // notice: circuitize_samples takes pqtrees, not ctrees!
//! let csamples = circuitize_samples::<Fr>(&samples, &pqtrees);
//! ```
//!
//! # Generating trees and samples for benchmarking
//!
//! Use the [`generate_trees_model`] and [`generate_samples`] functions (see above for example
//! usage).
//!
//! # Loading existing tree models & their samples
//!
//! E.g. to load tree models and samples resulting from the Python-based Catboost model processing
//! pipeline.
//!
//! ```
//! let trees_model: TreesModel = load_trees_model("src/zkdt/test_qtrees.json");
//! let samples: Vec<Vec<u16>> = load_samples("src/zkdt/test_samples_10x6.npy");
//! ```

extern crate serde;
extern crate serde_json;

use crate::zkdt::helpers::*;
use crate::zkdt::structs::{BinDecomp16Bit, DecisionNode, InputAttribute, LeafNode};
use crate::zkdt::trees::*;
use remainder_shared_types::FieldExt;
use ndarray::Array2;
use ndarray_npy::read_npy;
use serde::{Deserialize, Serialize};
use rand::Rng;
use std::fs::File;

/// The trees model resulting from the Python pipeline.
/// This struct is used for parsing JSON.
#[derive(Debug, Serialize, Deserialize)]
pub struct TreesModel {
    trees: Vec<Node<f64>>,
    bias: f64,
    scale: f64,
    n_features: usize,
}

/// Intermediate representation used for deriving CircuitizedTrees and CircuitizedSamples (given
/// samples to circuitize).
/// For properties, see PaddedQuantizedTrees.from().
pub struct PaddedQuantizedTrees {
    trees: Vec<Node<i32>>,
    depth: usize,
    scaling: f64,
}

/// Circuitized trees use flat (i.e. non-recursive) structs for the decision and leaf nodes and
/// represent all integers using the field.
/// Circuitized trees have the same properties as PaddedQuantizedTree, except that they include an
/// extra "dummy" decision node so that the number of decision nodes is a power of two (equal to
/// the number of leaf nodes).
/// The dummy decision node has node id 2^depth - 1.
pub struct CircuitizedTrees<F: FieldExt> {
    decision_nodes: Vec<Vec<DecisionNode<F>>>, // indexed by tree, then by node (sorted by node id)
    leaf_nodes: Vec<Vec<LeafNode<F>>>,         // indexed by tree, then by node (sorted by node id)
    depth: usize,
    scaling: f64,
}

type Sample<F> = Vec<InputAttribute<F>>;
/// Represents the circuitization of a batch of samples with respect to a PaddedQuantizedTrees.
/// Bit decompositions are little endian.
/// `differences` is a signed decomposition, with the sign bit at the end.
/// Each vector in `multiplicities` has length `2.pow(pqtrees.depth)`.
pub struct CircuitizedSamples<F: FieldExt> {
    samples: Vec<Sample<F>>,                        // indexed by samples
    permuted_samples: Vec<Vec<Sample<F>>>,          // indexed by trees, samples
    decision_paths: Vec<Vec<Vec<DecisionNode<F>>>>, // indexed by trees, samples, steps in path
    differences: Vec<Vec<Vec<BinDecomp16Bit<F>>>>,  // indexed by trees, samples, steps in path
    path_ends: Vec<Vec<LeafNode<F>>>,               // indexed by trees, samples
    multiplicities: Vec<Vec<BinDecomp16Bit<F>>>,    // indexed by trees, tree nodes
}

/// Circuitize the provided batch of samples using the specified PaddedQuantizedTrees instance,
/// returning a CircuitizedSamples instance.
/// See documentation of CircuitizedSamples.
/// The length of each returned Sample instance (both in `samples` and `permuted_samples`) is
/// `next_power_of_two((pqtrees.depth - 1) * values_array[0].len())`
/// Pre: `values_array.len() > 0`.
pub fn circuitize_samples<F: FieldExt>(
    values_array: &[Vec<u16>],
    pqtrees: &PaddedQuantizedTrees,
) -> CircuitizedSamples<F> {
    // repeat and pad the attributes of the sample
    let values_array: Vec<Vec<u16>> = values_array
        .iter()
        .map(|x| repeat_and_pad(x, pqtrees.depth - 1, 0_u16))
        .collect();
    let sample_length = values_array[0].len();

    let mut samples: Vec<Sample<F>> = vec![];
    let mut permuted_samples: Vec<Vec<Sample<F>>> = vec![];
    let mut decision_paths: Vec<Vec<Vec<DecisionNode<F>>>> = vec![];
    let mut path_ends: Vec<Vec<LeafNode<F>>> = vec![];
    let mut differences: Vec<Vec<Vec<BinDecomp16Bit<F>>>> = vec![];
    let mut multiplicities: Vec<Vec<BinDecomp16Bit<F>>> = vec![];

    // build the samples array
    for values in &values_array {
        samples.push(
            values
                .iter()
                .enumerate()
                .map(|(index, value)| InputAttribute {
                    attr_id: F::from(index as u64),
                    attr_val: F::from(*value as u64),
                })
                .collect(),
        );
    }

    for tree in &pqtrees.trees {
        // initialize the node visit counts "multiplicities"
        let mut multiplicities_for_tree: Vec<u32> = vec![0_u32; 2_usize.pow(pqtrees.depth as u32)];
        let mut permuted_samples_for_tree: Vec<Sample<F>> = vec![];
        let mut decision_paths_for_tree: Vec<Vec<DecisionNode<F>>> = vec![];
        let mut path_ends_for_tree: Vec<LeafNode<F>> = vec![];
        let mut differences_for_tree: Vec<Vec<BinDecomp16Bit<F>>> = vec![];

        for (values, sample) in values_array.iter().zip(samples.iter()) {
            // get the path
            let path = tree.get_path(&values);

            // derive data from decision path
            let mut decision_path = vec![];
            let mut permuted_sample: Sample<F> = vec![];
            let mut attribute_visits = vec![0; sample_length];
            let mut differences_for_tree_and_sample = vec![];
            for node in &path[..path.len() - 1] {
                if let Node::Internal {
                    id,
                    feature_index,
                    threshold,
                    ..
                } = node
                {
                    decision_path.push(DecisionNode {
                        node_id: F::from(id.unwrap() as u64),
                        attr_id: F::from(*feature_index as u64),
                        threshold: F::from(*threshold as u64),
                    });
                    // calculate the bit decompositions of the differences
                    let difference = (values[*feature_index] as i32) - (*threshold as i32);
                    differences_for_tree_and_sample
                        .push(build_signed_bit_decomposition::<F>(difference).unwrap());
                    // accumulate the multiplicities for this tree
                    multiplicities_for_tree[id.unwrap() as usize] += 1;
                    // build up the permuted sample
                    permuted_sample.push(sample[*feature_index]);
                    attribute_visits[*feature_index] += 1;
                } else {
                    panic!("All Nodes in the path must be internal, except the last");
                }
            }
            // populate the remaining attributes of the permuted sample
            for idx in 0..sample_length {
                if attribute_visits[idx] == 0 {
                    permuted_sample.push(sample[idx]);
                }
            }

            permuted_samples_for_tree.push(permuted_sample);
            decision_paths_for_tree.push(decision_path);
            differences_for_tree.push(differences_for_tree_and_sample);

            // build the leaf node
            if let Node::Leaf { id, value } = path[path.len() - 1] {
                path_ends_for_tree.push(LeafNode {
                    node_id: F::from(id.unwrap() as u64),
                    node_val: i32_to_field(*value),
                });
                // accumulate multiplicity for leaf node
                multiplicities_for_tree[id.unwrap() as usize] += 1;
            } else {
                panic!("Last item in path should be a Node::Leaf");
            }
        }
        permuted_samples.push(permuted_samples_for_tree);
        decision_paths.push(decision_paths_for_tree);
        path_ends.push(path_ends_for_tree);
        differences.push(differences_for_tree);
        // calculate the bit decompositions of the visit counts
        multiplicities.push(multiplicities_for_tree
                            .into_iter()
                            .map(build_unsigned_bit_decomposition)
                            .map(|option| option.unwrap())
                            .collect());
    }

    CircuitizedSamples {
        samples: samples,
        permuted_samples: permuted_samples,
        decision_paths: decision_paths,
        differences: differences,
        path_ends: path_ends,
        multiplicities: multiplicities,
    }
}

impl<F: FieldExt> From<&PaddedQuantizedTrees> for CircuitizedTrees<F> {
    /// Extract the DecisionNode and LeafNode instances from the PaddedQuantizedTrees instance to
    /// obtain a CircuitizedTrees.
    fn from(pqtrees: &PaddedQuantizedTrees) -> Self {
        // extract, sort & pad the decision nodes
        let mut decision_nodes = vec![];
        let dummy_node = DecisionNode {
            node_id: F::from(2_u64.pow(pqtrees.depth as u32) - 1),
            attr_id: F::from(0_u64),
            threshold: F::from(0_u64),
        };
        for tree in &pqtrees.trees {
            let mut tree_decision_nodes = extract_decision_nodes(&tree);
            tree_decision_nodes.sort_by_key(|node| node.node_id);
            // add a dummy node to make length a power of two
            tree_decision_nodes.push(dummy_node.clone());
            decision_nodes.push(tree_decision_nodes);
        }
        // extract and sort the leaf nodes
        let mut leaf_nodes = vec![];
        for tree in &pqtrees.trees {
            let mut tree_leaf_nodes = extract_leaf_nodes(&tree);
            tree_leaf_nodes.sort_by_key(|node| node.node_id);
            leaf_nodes.push(tree_leaf_nodes);
        }

        CircuitizedTrees {
            decision_nodes: decision_nodes,
            leaf_nodes: leaf_nodes,
            depth: pqtrees.depth,
            scaling: pqtrees.scaling,
        }
    }
}

impl From<&TreesModel> for PaddedQuantizedTrees {
    /// Given a TreesModel object representing a decision tree model operating on u32 samples, prepare the model for circuitization:
    /// 1. scale and bias are folded into the leaf values;
    /// 2. leaf values are symmetrically quantized to i32;
    /// 3. all trees are padded such that they are all perfect and of uniform depth (without modifying
    ///    the predictions of any tree) where the uniform depth is chosen to be 2^l + 1 for minimal l >= 0;
    /// 4. ids are assigned to all nodes (as per assign_id());
    /// 5. the feature indexes are transformed according to
    ///      idx -> (depth_of_node - 1) * trees_model.n_features + idx
    ///    thereby ensuring that each feature index occurs only once on each descent path.
    /// The resulting PaddedQuantizedTrees incorporates all the tree instances, the (uniform) depth
    /// of the trees, and the scaling factor to approximately undo the quantization (via division)
    /// after aggregating the scores.
    fn from(trees_model: &TreesModel) -> Self {
        let mut trees_f64 = trees_model.trees.clone();
        // fold scale into all trees
        for tree in &mut trees_f64 {
            tree.transform_values(&|value| trees_model.scale * value);
        }
        // fold bias into first tree
        trees_f64[0].transform_values(&|value| value + trees_model.bias);

        // quantize the leaf values
        let (qtrees, rescaling) = quantize_trees(&trees_f64);

        // pad the trees for perfection
        let max_depth = qtrees
            .iter()
            .map(|tree: &Node<i32>| tree.depth(std::cmp::max))
            .max()
            .unwrap();
        let target_depth = next_power_of_two(max_depth - 1).unwrap() + 1;
        // insert DecisionNodes where needed to perfect each tree
        let mut qtrees: Vec<Node<i32>> = qtrees
            .iter()
            .map(|tree: &Node<i32>| tree.perfect_to_depth(target_depth))
            .collect();
        // assign ids to all nodes
        for tree in &mut qtrees {
            tree.assign_id(0);
        }
        // transform feature indices such that they never repeat along a path
        for tree in &mut qtrees {
            tree.offset_feature_indices(trees_model.n_features);
        }

        PaddedQuantizedTrees {
            trees: qtrees,
            depth: target_depth,
            scaling: rescaling,
        }
    }
}

/// Generate a TreesModel as specified.  Meaning of arguments as per generate_trees().
/// Scale and bias are chosen randomly.
pub fn generate_trees_model(n_trees: usize, target_depth: usize, n_features: usize, premature_leaf_proba: f64) -> TreesModel {
    let mut rng = rand::thread_rng();
    TreesModel {
        trees: (0..n_trees)
            .map(|_| generate_tree(target_depth, n_features, premature_leaf_proba))
            .collect(),
        bias: rng.gen(),
        scale: rng.gen(),
        n_features: n_features
    }
}

/// Generate an array of samples for input into the trees model.
/// Values less than or equal to [`SIGNED_DECOMPOSITION_MAX_ARG_ABS`] (for the benefit of
/// [`build_signed_bit_decomposition`]).
pub fn generate_samples(n_samples: usize, n_features: usize) -> Vec<Vec<u16>> {
    let mut rng = rand::thread_rng();
    (0..n_samples)
        .map(|_| (0..n_features).map(|_| rng.gen_range(0..(SIGNED_DECOMPOSITION_MAX_ARG_ABS + 1) as u16)).collect())
        .collect()
}

/// Load a 2d array of samples from a `.npy` file.
/// Pre: all values are less than or equal to [`SIGNED_DECOMPOSITION_MAX_ARG_ABS`] (for the benefit of
/// [`build_signed_bit_decomposition`]).
pub fn load_samples(filename: &str) -> Vec<Vec<u16>> {
    let input_arr: Array2<u16> = read_npy(filename).unwrap();
    // check sample values are in bounds
    assert!(input_arr.iter().all(|&x| x <= SIGNED_DECOMPOSITION_MAX_ARG_ABS as u16));
    input_arr.outer_iter().map(|row| row.to_vec()).collect()
}

/// Load a trees model from a JSON file.
/// Pre: all threshold values are less than or equal to [`SIGNED_DECOMPOSITION_MAX_ARG_ABS`] (for the benefit of
/// [`build_signed_bit_decomposition`]).
pub fn load_trees_model(filename: &str) -> TreesModel {
    let file = File::open(filename)
        .expect(&format!("'{}' should be available.", filename));
    let trees_model: TreesModel = serde_json::from_reader(file).expect(&format!(
        "'{}' should be valid TreesModel JSON.",
        filename
    ));
    // FIXME check the thresholds of every node on every tree
    trees_model
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;

    /// Returns a small tree for testing:
    ///      .
    ///     / \
    ///    .  1.2
    ///   / \
    /// 0.1 0.2
    fn build_small_tree() -> Node<f64> {
        let left = Node::new_leaf(None, 0.1);
        let middle = Node::new_leaf(None, 0.2);
        let right = Node::new_leaf(None, 1.2);
        let internal = Node::new_internal(None, 0, 2, left, middle);
        Node::new_internal(None, 1, 1, internal, right)
    }

    #[test]
    fn test_trees_model_loading() {
        let filename = "src/zkdt/test_qtrees.json";
        let trees_model = load_trees_model(filename);
    }

    #[test]
    fn test_numpy_loading() {
        let filename = "src/zkdt/test_samples_10x6.npy";
        let samples = load_samples(filename);
        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn test_circuitize_samples() {
        let sample_length = 5;
        let samples = vec![
            vec![0_u16; sample_length],
            vec![2_u16, 0_u16, 0_u16, 0_u16, 0_u16],
            vec![2_u16; sample_length],
        ];
        let tree = build_small_tree();
        let trees_model = TreesModel {
            trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
            bias: 1.1,
            scale: 6.6,
            n_features: sample_length,
        };
        let pqtrees: PaddedQuantizedTrees = (&trees_model).into();
        let csamples = circuitize_samples::<Fr>(&samples, &pqtrees);
        // check size of outer dimensions
        let n_trees = trees_model.trees.len();
        let repeated_sample_length =
            next_power_of_two((pqtrees.depth - 1) * sample_length).unwrap();
        assert_eq!(csamples.samples.len(), samples.len());

        // check dimensions of permuted samples
        assert_eq!(csamples.permuted_samples.len(), n_trees);
        for permuted_samples_for_tree in &csamples.permuted_samples {
            assert_eq!(permuted_samples_for_tree.len(), samples.len());
            for permuted_sample in permuted_samples_for_tree {
                assert_eq!(permuted_sample.len(), repeated_sample_length);
            }
        }
        // check the contents of the permuted samples for the non-trivial tree (the 0th)
        let permuted_sample = &csamples.permuted_samples[0][0];
        assert_eq!(permuted_sample[0].attr_id, Fr::from(1));
        // ... sample travels left down the tree
        assert_eq!(
            permuted_sample[1].attr_id,
            Fr::from(sample_length as u64 + 0)
        );

        // check the dimension of the decision paths
        assert_eq!(csamples.decision_paths.len(), n_trees);
        for decision_paths_for_tree in &csamples.decision_paths {
            assert_eq!(decision_paths_for_tree.len(), samples.len());
            for decision_path in decision_paths_for_tree {
                assert_eq!(decision_path.len(), pqtrees.depth - 1);
            }
        }
        // check decision path contents for one combination
        let decision_path = &csamples.decision_paths[0][2];
        assert_eq!(decision_path[0].node_id, Fr::from(0));
        // ... sample travels right down the tree
        assert_eq!(decision_path[1].node_id, Fr::from(2));

        // check the dimension of the path_ends
        assert_eq!(csamples.path_ends.len(), n_trees);
        for path_ends_for_tree in &csamples.path_ends {
            assert_eq!(path_ends_for_tree.len(), samples.len());
        }
        // check the contents
        let path_ends_for_tree = &csamples.path_ends[0];
        assert_eq!(path_ends_for_tree[0].node_id, Fr::from(3));
        assert_eq!(path_ends_for_tree[1].node_id, Fr::from(4));

        // check the dimensions of the differences
        assert_eq!(csamples.differences.len(), n_trees);
        for differences_for_tree in &csamples.differences {
            assert_eq!(differences_for_tree.len(), samples.len());
            for differences in differences_for_tree {
                assert_eq!(differences.len(), pqtrees.depth - 1);
            }
        }
        // check contents
        let differences = &csamples.differences[0][0];
        // should have the bit decomposition of -2
        assert_eq!(differences[1].bits[0], Fr::from(0));
        assert_eq!(differences[1].bits[1], Fr::from(1));
        assert_eq!(differences[1].bits[2], Fr::from(0));
        assert_eq!(differences[1].bits[15], Fr::from(1));

        // check the multiplicities
        let multiplicities = csamples.multiplicities;
        let n_nodes = 2_usize.pow(pqtrees.depth as u32); // includes dummy node
        assert_eq!(multiplicities.len(), n_trees);
        for multiplicities_for_tree in &multiplicities {
            assert_eq!(multiplicities_for_tree.len(), n_nodes);
            // root node id has multiplicity samples.len() = 3
            let multiplicity = &multiplicities_for_tree[0];
            assert_eq!(multiplicity.bits[0], Fr::from(1));
            assert_eq!(multiplicity.bits[1], Fr::from(1));
            assert_eq!(multiplicity.bits[2], Fr::from(0));
            // dummy node id has multiplicity 0
            let multiplicity = &multiplicities_for_tree[n_nodes - 1];
            for bit in multiplicity.bits {
                assert_eq!(bit, Fr::from(0));
            }
        }
    }

    #[test]
    fn test_padded_quantized_trees_from() {
        let tree = build_small_tree();
        let trees_model = TreesModel {
            trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
            bias: 1.1,
            scale: 6.6,
            n_features: 11,
        };
        let pqtrees: PaddedQuantizedTrees = (&trees_model).into();
        assert_eq!(pqtrees.trees.len(), 2);
        assert_eq!(pqtrees.depth, 3);
        // check trees are as claimed
        for tree in &pqtrees.trees {
            assert_eq!(tree.depth(std::cmp::max), 3);
            assert!(tree.is_perfect());
            assert_eq!(tree.get_id().unwrap(), 0);
        }
    }

    #[test]
    fn test_circuitized_trees_from() {
        let tree = build_small_tree();
        let trees_model = TreesModel {
            trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
            bias: 1.1,
            scale: 6.6,
            n_features: 11,
        };
        let pqtrees: PaddedQuantizedTrees = (&trees_model).into();
        let ctrees: CircuitizedTrees<Fr> = (&pqtrees).into();
        assert_eq!(ctrees.depth, pqtrees.depth);
        // check decision nodes
        for tree_dns in &ctrees.decision_nodes {
            assert_eq!(tree_dns.len(), 4);
            for (node_id, node) in tree_dns.iter().take(tree_dns.len() - 1).enumerate() {
                assert_eq!(node.node_id, Fr::from(node_id as u64));
            }
            assert_eq!(tree_dns[tree_dns.len() - 1].node_id, Fr::from(7 as u64));
        }
        // check leaf nodes
        for tree_lns in &ctrees.leaf_nodes {
            assert_eq!(tree_lns.len(), 4);
            for (node_id, node) in tree_lns.iter().enumerate() {
                assert_eq!(node.node_id, Fr::from((4 - 1) + node_id as u64));
            }
        }
        // accumulate score by taking the value of the first leaf node
        let mut acc_score: Fr = Fr::from(0);
        acc_score += ctrees.leaf_nodes[0][0].node_val;
        acc_score += ctrees.leaf_nodes[1][0].node_val;
        // check that the quantized scores accumulated as expected
        let expected_score = trees_model.scale * (0.1 + 3.0) + trees_model.bias;
        let quant_score = (expected_score * ctrees.scaling) as i32;
        let f_quant_score = if quant_score >= 0 {
            Fr::from(quant_score as u64)
        } else {
            -Fr::from(quant_score.abs() as u64)
        };
        // just check that's it's close
        assert_eq!(f_quant_score, acc_score + Fr::from(1));
    }

    #[test]
    fn test_generate_trees_model() {
        let target_depth = 3;
        let n_features = 6;
        let n_trees = 11;
        let trees_model = generate_trees_model(n_trees, target_depth, n_features, 0.5);
        assert_eq!(trees_model.n_features, n_features);
        assert_eq!(trees_model.trees.len(), n_trees);
        for tree in &trees_model.trees {
            assert!(tree.depth(std::cmp::max) <= target_depth);
        }
    }

    #[test]
    fn test_generate_samples() {
        let samples = generate_samples(10, 3);
        assert_eq!(samples.len(), 10);
        for sample in &samples {
            assert_eq!(sample.len(), 3);
        }
    }

    #[test]
    fn test_documentation() { // TODO remove once the doctests are being run
        let n_trees = 3;
        let n_features = 4;
        let depth = 6;
        // start with some random (probably not perfect) trees with f64 leaf values
        let trees_model = generate_trees_model(n_trees, depth, n_features, 0.5);
        // pad the trees, assign ids and quantize the leaf values
        let pqtrees: PaddedQuantizedTrees = (&trees_model).into();
        // circuitize the trees (converts to DecisionNode<F>, LeafNode<F>)
        let ctrees: CircuitizedTrees<Fr> = (&pqtrees).into();
        // .. continued
        // generate some samples to play with
        let n_samples = 10;
        let samples = generate_samples(n_samples, n_features);
        // notice: circuitize_samples takes pqtrees, not ctrees!
        let csamples = circuitize_samples::<Fr>(&samples, &pqtrees);
        // .. continued
        let trees_model: TreesModel = load_trees_model("src/zkdt/test_qtrees.json");
        let samples: Vec<Vec<u16>> = load_samples("src/zkdt/test_samples_10x6.npy");
    }
}