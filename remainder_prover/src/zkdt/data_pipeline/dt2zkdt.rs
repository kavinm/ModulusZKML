//! Conversion from a decision trees model to a circuit ready form.
//!
//! # Circuitizing tree models
//! (c.f. [`CircuitizedTrees`])
//!
//! ```
//! use ark_bn254::Fr;
//!
//! let n_trees = 3;
//! let n_features = 4;
//! let depth = 6;
//! // start with some random (probably not perfect) trees with f64 leaf values
//! let raw_trees_model = generate_raw_trees_model(n_trees, depth, n_features, 0.5);
//! // pad the trees, assign ids and quantize the leaf values
//! let trees_model: TreesModel = (&raw_trees_model).into();
//! // circuitize the trees (converts to DecisionNode<F>, LeafNode<F>)
//! let ctrees: CircuitizedTrees<Fr> = (&trees_model).into();
//! ```
//!
//! # Circuitizing samples
//! (c.f. [`CircuitizedSamples`]). Continuing the above example:
//!
//! ```
//! let n_samples = 10;
//! let raw_samples = generate_raw_samples(n_samples, n_features);
//! let samples: Samples = (&raw_samples).into();
//! let _csamples: CircuitizedSamples<Fr> = (&samples).into();
//! ```
//!
//! # Building auxiliary inputs
//! (c.f. [`CircuitizedAuxiliaries`]). Continuing the above example:
//!
//! ```
//! let _caux = circuitize_auxiliaries::<Fr>(&samples, &trees_model);
//! ```
//!
//! # Generating trees and samples for benchmarking
//!
//! Use the [`generate_raw_trees_model`] and [`generate_raw_samples`] functions (see above for example
//! usage).
//!
//! # Loading existing tree models & their samples
//!
//! E.g. to load tree models and samples resulting from the Python-based Catboost model processing
//! pipeline.
//!
//! ```
//! let raw_trees_model: RawTreesModel = load_raw_trees_model(Path::new("src/zkdt/data_pipeline/test_qtrees.json"));
//! let raw_samples: RawSamples = load_raw_samples(Path::new("src/zkdt/data_pipeline/test_samples_10x6.npy"));
//! ```

extern crate serde;
extern crate serde_json;

use crate::zkdt::data_pipeline::trees::*;
use crate::zkdt::structs::{
    BinDecomp16Bit, BinDecomp4Bit, BinDecomp8Bit, DecisionNode, InputAttribute, LeafNode,
};

use ark_serialize::Read;
use ndarray::Array2;
use ndarray_npy::read_npy;
use rand::Rng;
use rayon::prelude::*;
use remainder_shared_types::{FieldExt, Fr};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::File;
use std::path::Path;

use super::helpers::{
    build_signed_bit_decomposition, build_unsigned_bit_decomposition, extract_decision_nodes,
    extract_leaf_nodes, i64_to_field, next_power_of_two,
};

/// The trees model resulting from the Python pipeline.
/// This struct is used for parsing JSON.
#[derive(Debug, Serialize, Deserialize)]
pub struct RawTreesModel {
    trees: Vec<Node<f64>>,
    bias: f64,
    scale: f64,
    n_features: usize,
}

/// Used for deriving CircuitizedTrees and CircuitizedAuxiliaries (given samples).
/// For properties, see TreesModel.from().
pub struct TreesModel {
    trees: Vec<Node<i64>>,
    depth: usize,
    scaling: f64,
}

/// Circuitized trees use flat (i.e. non-recursive) structs for the decision and leaf nodes and
/// represent all integers using the field.
/// Circuitized trees have the same properties as TreesModel, except that they include an
/// extra "dummy" decision node so that the number of decision nodes is a power of two (equal to
/// the number of leaf nodes).
/// The dummy decision node has node id 2^depth - 1.
pub struct CircuitizedTrees<F: FieldExt> {
    pub decision_nodes: Vec<Vec<DecisionNode<F>>>, // indexed by tree, then by node (sorted by node id)
    pub leaf_nodes: Vec<Vec<LeafNode<F>>>, // indexed by tree, then by node (sorted by node id)
    pub depth: usize,
    pub scaling: f64,
}

/// Output of load_raw_samples, for conversion to a Samples instance.
/// Pre: Values do not exceed a 15-bit (not 16 bit!) representation.
#[derive(Clone)]
pub struct RawSamples {
    pub values: Vec<Vec<u16>>,
    pub sample_length: usize,
}

/// Difference to RawSamples: each sample is padded length-wise to the next power of two, and
/// the number of samples is also padded to be the next power of two.
pub struct Samples {
    pub values: Vec<Vec<u16>>,
    pub sample_length: usize,
}

/// Represents the circuitization of a batch of samples.
pub type CircuitizedSamples<F: FieldExt> = Vec<Vec<InputAttribute<F>>>;

/// Represents the circuitization of a batch of samples with respect to a TreesModel.
/// * Bit decompositions are little endian.
/// * `differences` is a signed decomposition, with the sign bit at the end.
/// * Each vector in `node_multiplicities` has length `2.pow(trees_model.depth)`; it is indexed by
/// node_id for decision nodes, and by node id + 1 for leaf nodes.
pub struct CircuitizedAuxiliaries<F: FieldExt> {
    pub decision_paths: Vec<Vec<Vec<DecisionNode<F>>>>, // indexed by trees, samples, steps in path
    pub attributes_on_paths: Vec<Vec<Vec<InputAttribute<F>>>>, // indexed by trees, samples, steps in path
    pub differences: Vec<Vec<Vec<BinDecomp16Bit<F>>>>, // indexed by trees, samples, steps in path
    pub path_ends: Vec<Vec<LeafNode<F>>>,              // indexed by trees, samples
    pub attribute_multiplicities: Vec<Vec<Vec<BinDecomp4Bit<F>>>>, // indexed by trees, samples, then by attribute index FIXME TO BE REMOVED (@Ben)
    pub attribute_multiplicities_per_sample: Vec<Vec<BinDecomp8Bit<F>>>, // indexed by samples, then by attribute index (so aggregated over trees)
    pub node_multiplicities: Vec<Vec<BinDecomp16Bit<F>>>, // indexed by trees, tree nodes
}

impl fmt::Display for RawSamples {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{:?} samples with {:?} attributes",
            self.values.len(),
            self.sample_length
        )
    }
}

impl RawSamples {
    pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
        let values = self.values[range].to_vec();
        RawSamples {
            values: values,
            sample_length: self.sample_length,
        }
    }
}

impl From<&RawSamples> for Samples {
    /// Prepare the provided RawSamples for processing by the TreesModel by padding the raw sample
    /// values such that the length of each individual sample is a power of two, and that the number of
    /// samples is also a power of two.
    /// Pre: raw_samples.values.len() > 0.
    fn from(raw_samples: &RawSamples) -> Self {
        let mut samples: Vec<Vec<u16>> = vec![];
        let sample_length = next_power_of_two(raw_samples.values[0].len()).unwrap();
        for raw_sample in &raw_samples.values {
            let mut sample = raw_sample.clone();
            sample.resize(sample_length, 0);
            samples.push(sample);
        }
        let target_sample_count = next_power_of_two(raw_samples.values.len()).unwrap();
        for i in raw_samples.values.len()..target_sample_count {
            samples.push(vec![0_u16; sample_length]);
        }
        Samples {
            values: samples,
            sample_length,
        }
    }
}

/// Conversion TreePath -> LeafNode
impl<F: FieldExt> From<&TreePath<i64>> for LeafNode<F> {
    fn from(tree_path: &TreePath<i64>) -> Self {
        LeafNode {
            node_id: F::from(tree_path.leaf_node_id as u64),
            node_val: i64_to_field(tree_path.leaf_value),
        }
    }
}

impl<F: FieldExt> From<&Samples> for CircuitizedSamples<F> {
    fn from(samples: &Samples) -> Self {
        samples
            .values
            .par_iter()
            .map(build_sample_witness)
            .collect()
    }
}

type DecisionPath<F> = Vec<DecisionNode<F>>;
impl<F: FieldExt, T: Copy> From<&TreePath<T>> for DecisionPath<F> {
    fn from(tree_path: &TreePath<T>) -> Self {
        tree_path
            .path_steps
            .iter()
            .map(|path_step| DecisionNode::<F> {
                node_id: F::from(path_step.node_id as u64),
                attr_id: F::from(path_step.feature_index as u64),
                threshold: F::from(path_step.threshold as u64),
            })
            .collect()
    }
}

type AttributesOnPath<F> = Vec<InputAttribute<F>>;
// conversion from tree path to attributes on path
impl<F: FieldExt> From<&TreePath<i64>> for AttributesOnPath<F> {
    fn from(tree_path: &TreePath<i64>) -> Self {
        tree_path
            .path_steps
            .iter()
            .map(|step| InputAttribute::<F> {
                attr_id: F::from(step.feature_index as u64),
                attr_val: F::from(step.feature_value as u64),
            })
            .collect()
    }
}

type DifferencesBits<F> = Vec<BinDecomp16Bit<F>>;
impl<F: FieldExt> From<&TreePath<i64>> for DifferencesBits<F> {
    fn from(tree_path: &TreePath<i64>) -> Self {
        tree_path
            .path_steps
            .iter()
            .map(|step| {
                let difference = (step.feature_value as i32) - (step.threshold as i32);
                build_differences_bindecomp(difference)
            })
            .collect()
    }
}

fn build_differences_bindecomp<F: FieldExt>(difference: i32) -> BinDecomp16Bit<F> {
    let bits = build_signed_bit_decomposition(difference, 16).unwrap();
    BinDecomp16Bit::<F>::from(bits)
}

fn build_node_multiplicity_bindecomp<F: FieldExt>(multiplicity: usize) -> BinDecomp16Bit<F> {
    let bits = build_unsigned_bit_decomposition(multiplicity as u32, 16).unwrap();
    BinDecomp16Bit::<F>::from(bits)
}

/// FIXME TO BE REMOVED (@Ben)
fn build_4bit_attribute_multiplicity_bindecomp<F: FieldExt>(
    multiplicity: usize,
) -> BinDecomp4Bit<F> {
    let bits = build_unsigned_bit_decomposition(multiplicity as u32, 4).unwrap();
    BinDecomp4Bit::<F>::from(bits)
}

fn build_8bit_attribute_multiplicity_bindecomp<F: FieldExt>(
    multiplicity: usize,
) -> BinDecomp8Bit<F> {
    let bits = build_unsigned_bit_decomposition(multiplicity as u32, 8).unwrap();
    //  assert_eq!(recomp_8bit(&bits), multiplicity);
    let recomp = recomp_8bit(&bits);
    if recomp != multiplicity {
        dbg!("Not equal: {}, {}", &bits, &multiplicity);
        dbg!(multiplicity);
        panic!();
    }
    BinDecomp8Bit::<F>::from(bits)
}

/// Testing/sanitycheck
fn recomp_8bit(bits: &Vec<bool>) -> usize {
    assert_eq!(bits.len(), 8);
    bits.into_iter().enumerate().fold(0, |acc, (idx, bit)| {
        acc + ((1 << idx) * (if *bit { 1 } else { 0 }))
    })
}

/// Build the witnesses for a single sample.
fn build_sample_witness<F: FieldExt>(sample: &Vec<u16>) -> Vec<InputAttribute<F>> {
    sample
        .iter()
        .enumerate()
        .map(|(index, value)| InputAttribute {
            attr_id: F::from(index as u64),
            attr_val: F::from(*value as u64),
        })
        .collect()
}

/// Circuitize the provided batch of samples using the specified TreesModel instance,
/// returning a CircuitizedAuxiliaries instance.
/// See documentation of [`CircuitizedAuxiliaries`].
/// The following preconditions are required to ensure attribute multiplicities fit in 8 bit bindecomp:
/// Pre: trees_model.trees.len() <= 2^5
/// Pre: trees_model.depth <= 2^3 + 1
pub fn circuitize_auxiliaries<F: FieldExt>(
    samples_in: &Samples,
    trees_model: &TreesModel,
) -> CircuitizedAuxiliaries<F> {
    //
    assert!(trees_model.trees.len() <= 32);
    assert!(trees_model.depth <= 9);
    let paths: Vec<Vec<TreePath<i64>>> = trees_model
        .trees
        .par_iter()
        .map(|tree| {
            samples_in
                .values
                .par_iter()
                .map(|sample| tree.get_tree_path(&sample))
                .collect()
        })
        .collect();

    let decision_paths: Vec<Vec<DecisionPath<F>>> = paths
        .par_iter()
        .map(|tree_paths| tree_paths.par_iter().map(DecisionPath::<F>::from).collect())
        .collect();

    let attributes_on_paths: Vec<Vec<AttributesOnPath<F>>> = paths
        .par_iter()
        .map(|tree_paths| {
            tree_paths
                .par_iter()
                .map(AttributesOnPath::<F>::from)
                .collect()
        })
        .collect();

    let differences: Vec<Vec<DifferencesBits<F>>> = paths
        .par_iter()
        .map(|tree_paths| {
            tree_paths
                .par_iter()
                .map(DifferencesBits::<F>::from)
                .collect()
        })
        .collect();

    let path_ends: Vec<Vec<LeafNode<F>>> = paths
        .par_iter()
        .map(|tree_paths| tree_paths.iter().map(LeafNode::<F>::from).collect())
        .collect();

    let attribute_multiplicities: Vec<Vec<Vec<BinDecomp4Bit<F>>>> = paths
        .par_iter()
        .map(|tree_paths| {
            tree_paths
                .iter()
                .map(|tree_path| {
                    count_attribute_multiplicities(&tree_path.path_steps, samples_in.sample_length)
                })
                .into_iter()
                .map(|multiplicities| {
                    multiplicities
                        .into_iter()
                        .map(build_4bit_attribute_multiplicity_bindecomp)
                        .collect()
                })
                .collect()
        })
        .collect();

    // BUILD THE ATTRIBUTE MULTIPLICITIES, AGGD OVER TREES
    // check that each tree has one path per sample ...
    let sample_count: usize = samples_in.values.len();
    assert!(paths
        .iter()
        .all(|paths_for_tree| paths_for_tree.len() == sample_count));
    // ... so that the nesting can be reversed, to become: samples first, then trees
    let _paths_per_sample: Vec<Vec<TreePath<i64>>> = (0..sample_count)
        .map(|sample_idx| {
            paths
                .iter()
                .filter_map(|paths_for_tree| paths_for_tree.get(sample_idx))
                .cloned()
                .collect()
        })
        .collect();

    let attribute_multiplicities_per_sample: Vec<Vec<BinDecomp8Bit<F>>> = _paths_per_sample
        .par_iter()
        .map(|paths_of_sample| {
            let mut multiplicities = vec![0_usize; samples_in.sample_length];
            paths_of_sample.iter().for_each(|path_of_sample_thru_tree| {
                path_of_sample_thru_tree
                    .path_steps
                    .iter()
                    .for_each(|step| multiplicities[step.feature_index as usize] += 1);
            });
            multiplicities
        })
        .map(|multiplicities| {
            multiplicities
                .into_iter()
                .map(build_8bit_attribute_multiplicity_bindecomp)
                .collect()
        })
        .collect();

    let node_multiplicities: Vec<Vec<BinDecomp16Bit<F>>> = paths
        .par_iter()
        .map(|tree_paths| {
            count_node_multiplicities(tree_paths, trees_model.depth)
                .into_par_iter()
                .map(build_node_multiplicity_bindecomp)
                .collect()
        })
        .collect();

    CircuitizedAuxiliaries {
        decision_paths,
        attributes_on_paths,
        differences,
        path_ends,
        attribute_multiplicities,
        attribute_multiplicities_per_sample,
        node_multiplicities,
    }
}

/// FIXME TO BE REMOVED (@Ben)
/// Given a vector of instances of [`PathStep`], return a vector counting how often each attribute
/// is used.
pub fn count_attribute_multiplicities(
    path_steps: &Vec<PathStep>,
    sample_length: usize,
) -> Vec<usize> {
    let mut multiplicities = vec![0_usize; sample_length];
    path_steps
        .iter()
        .for_each(|step| multiplicities[step.feature_index as usize] += 1);
    multiplicities
}

/// Given a vector of instances of [`TreePath`] all belonging to the same tree, return a vector
/// counting the number of times each node was visited.
/// **WARNING**: Vector is indexed by node id for internal nodes, and by node_id + 1 for leaf nodes.
pub fn count_node_multiplicities<T: Copy>(
    tree_paths: &Vec<TreePath<T>>,
    tree_depth: usize,
) -> Vec<usize> {
    let mut multiplicities = vec![0_usize; 2_usize.pow(tree_depth as u32)];
    tree_paths.iter().for_each(|tree_path| {
        // count visits to internal nodes
        tree_path
            .path_steps
            .iter()
            .for_each(|step| multiplicities[step.node_id as usize] += 1);
        // count visits to leaf nodes
        multiplicities[tree_path.leaf_node_id as usize + 1] += 1;
    });
    multiplicities
}

impl<F: FieldExt> From<&TreesModel> for CircuitizedTrees<F> {
    /// Extract the DecisionNode and LeafNode instances from the TreesModel instance to
    /// obtain a CircuitizedTrees.
    fn from(trees_model: &TreesModel) -> Self {
        // extract, sort & pad the decision nodes
        let mut decision_nodes = vec![];
        let dummy_node = DecisionNode {
            node_id: F::from(2_u64.pow(trees_model.depth as u32) - 1),
            attr_id: F::from(0_u64),
            threshold: F::from(0_u64),
        };
        for tree in &trees_model.trees {
            let mut tree_decision_nodes = extract_decision_nodes(tree);
            tree_decision_nodes.sort_by_key(|node| node.node_id);
            // add a dummy node to make length a power of two
            tree_decision_nodes.push(dummy_node);
            decision_nodes.push(tree_decision_nodes);
        }
        // extract and sort the leaf nodes
        let mut leaf_nodes = vec![];
        for tree in &trees_model.trees {
            let mut tree_leaf_nodes = extract_leaf_nodes(tree);
            tree_leaf_nodes.sort_by_key(|node| node.node_id);
            leaf_nodes.push(tree_leaf_nodes);
        }

        CircuitizedTrees {
            decision_nodes,
            leaf_nodes,
            depth: trees_model.depth,
            scaling: trees_model.scaling,
        }
    }
}

impl From<&RawTreesModel> for TreesModel {
    /// Given a RawTreesModel object representing a decision tree model operating on u32 samples, prepare the model for circuitization:
    /// 1. scale and bias are folded into the leaf values;
    /// 2. leaf values are symmetrically quantized to i64;
    /// 3. all trees are padded such that they are all perfect and of uniform depth (without modifying
    ///    the predictions of any tree) where the uniform depth is chosen to be 2^l + 1 for minimal l >= 0;
    /// 4. ids are assigned to all nodes (as per assign_id());
    /// The resulting TreesModel incorporates all the tree instances, the (uniform) depth
    /// of the trees, and the scaling factor to approximately undo the quantization (via division)
    /// after aggregating the scores.
    fn from(raw_trees_model: &RawTreesModel) -> Self {
        let mut trees_f64 = raw_trees_model.trees.clone();
        // fold scale into all trees
        for tree in &mut trees_f64 {
            tree.transform_values(&|value| raw_trees_model.scale * value);
        }
        // fold bias into first tree
        trees_f64[0].transform_values(&|value| value + raw_trees_model.bias);

        // quantize the leaf values
        let (qtrees, rescaling) = quantize_leaf_values(&trees_f64);

        // pad the trees for perfection
        let max_depth = qtrees
            .iter()
            .map(|tree: &Node<i64>| tree.depth(std::cmp::max))
            .max()
            .unwrap();
        let target_depth = next_power_of_two(max_depth - 1).unwrap() + 1;
        // insert decision nodes where needed to perfect each tree
        let mut qtrees: Vec<Node<i64>> = qtrees
            .iter()
            .map(|tree: &Node<i64>| tree.perfect_to_depth(target_depth))
            .collect();
        // assign ids to all nodes
        for tree in &mut qtrees {
            tree.assign_id(0);
        }

        TreesModel {
            trees: qtrees,
            depth: target_depth,
            scaling: rescaling,
        }
    }
}

impl fmt::Display for RawTreesModel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let max_depth = self
            .trees
            .iter()
            .map(|tree: &Node<f64>| tree.depth(std::cmp::max))
            .max()
            .unwrap();
        write!(
            f,
            "{:?} trees of maximum depth {:?}",
            self.trees.len(),
            max_depth
        )
    }
}

impl RawTreesModel {
    pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
        let trees = self.trees[range].to_vec();
        RawTreesModel {
            trees: trees,
            bias: self.bias,
            scale: self.scale,
            n_features: self.n_features,
        }
    }
}

impl TreesModel {
    pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
        let trees = self.trees[range].to_vec();
        TreesModel {
            trees: trees,
            depth: self.depth,
            scaling: self.scaling,
        }
    }

    /// Pad (in-place) the trees to a multiple of the provided number.
    /// Padding trees are perfect of the same depth as existing trees, use decision nodes with DUMMY_FEATURE_INDEX and DUMMY_THRESHOLD, have leaf values of 0_i64, have ids assigned, and are appended to the end of self.trees.
    /// Pre: multiple > 0.
    pub fn pad_tree_count_to_multiple(&mut self, multiple: usize) {
        assert!(multiple > 0);
        if self.trees.len() % multiple == 0 {
            return;
        }
        // number of new trees required
        let shortfall = multiple - (self.trees.len() % multiple);
        //let trees: Vec<Node<i64>> = self
        self.trees = self
            .trees
            .iter()
            .cloned()
            .chain((0..shortfall).map(|_| {
                let mut tree = Node::new_constant_tree(
                    self.depth,
                    Node::<i64>::DUMMY_FEATURE_INDEX,
                    Node::<i64>::DUMMY_THRESHOLD,
                    0_i64,
                );
                tree.assign_id(0);
                tree
            }))
            .collect();
    }
}

/// Generate a RawTreesModel as specified.  Meaning of arguments as per [`generate_tree`].
/// Scale and bias are chosen randomly.
pub fn generate_raw_trees_model(
    n_trees: usize,
    target_depth: usize,
    n_features: usize,
    premature_leaf_proba: f64,
) -> RawTreesModel {
    let mut rng = rand::thread_rng();
    RawTreesModel {
        trees: (0..n_trees)
            .map(|_| generate_tree(target_depth, n_features, premature_leaf_proba))
            .collect(),
        bias: rng.gen(),
        scale: rng.gen(),
        n_features,
    }
}

/// Generate an array of samples for input into the trees model.  For demonstration purposes.
pub fn generate_raw_samples(n_samples: usize, n_features: usize) -> RawSamples {
    let mut rng = rand::thread_rng();
    let values = (0..n_samples)
        .map(|_| {
            (0..n_features)
                .map(|_| rng.gen_range(0..(2_u32.pow(15) + 1) as u16))
                .collect()
        })
        .collect();
    RawSamples {
        values,
        sample_length: n_features,
    }
}

/// Load a 2d array of samples from a `.npy` file.
pub fn load_raw_samples(filename: &Path) -> RawSamples {
    let input_arr: Array2<u16> = read_npy(filename).unwrap();
    RawSamples {
        values: input_arr.outer_iter().map(|row| row.to_vec()).collect(),
        sample_length: input_arr.shape()[1],
    }
}

/// Load a trees model from a JSON file.
/// WARNING: note the pre-condition.  No checks are performed.
/// Pre: all threshold values fit in a 16 bit signed bit decomposition.
pub fn load_raw_trees_model(filename: &Path) -> RawTreesModel {
    let mut file = File::open(filename)
        .unwrap_or_else(|_| panic!("'{:?}' should be available.", filename.to_path_buf()));

    let initial_buffer_size = file.metadata().map(|m| m.len() as usize + 1).unwrap_or(0);
    let mut bufreader = Vec::with_capacity(initial_buffer_size);
    file.read_to_end(&mut bufreader).unwrap();

    let raw_trees_model: RawTreesModel =
        serde_json::from_slice(&bufreader[..]).unwrap_or_else(|_| {
            panic!(
                "'{:?}' should be valid RawTreesModel JSON.",
                filename.to_path_buf()
            )
        });

    raw_trees_model
}

#[cfg(test)]
mod tests {
    use super::*;
    use remainder_shared_types::Fr;

    /// Returns a small tree for testing:
    ///       x[1]>=1?
    ///        /    \
    ///    x[0]>=2?  1.2
    ///     /    \
    ///  0.1      0.2
    fn build_small_tree() -> Node<f64> {
        let left = Node::new_leaf(None, 0.1);
        let middle = Node::new_leaf(None, 0.2);
        let right = Node::new_leaf(None, 1.2);
        let internal = Node::new_internal(None, 0, 2, left, middle);
        Node::new_internal(None, 1, 1, internal, right)
    }

    /// Returns another small tree for testing:
    ///       x[3]>=1?
    ///        /    \
    ///    x[4]>=2?  1.2
    ///     /    \
    ///  0.1      0.2
    fn build_small_tree_variant() -> Node<f64> {
        let left = Node::new_leaf(None, 0.1);
        let middle = Node::new_leaf(None, 0.2);
        let right = Node::new_leaf(None, 1.2);
        let internal = Node::new_internal(None, 4, 2, left, middle);
        Node::new_internal(None, 3, 1, internal, right)
    }

    #[test]
    fn test_raw_trees_model_loading() {
        let filename = "src/zkdt/data_pipeline/test_qtrees.json";
        let _raw_trees_model = load_raw_trees_model(Path::new(filename));
    }

    #[test]
    fn test_numpy_loading() {
        let filename = "src/zkdt/data_pipeline/test_samples_10x6.npy";
        let raw_samples = load_raw_samples(Path::new(filename));
        assert_eq!(raw_samples.values.len(), 10);
    }

    #[test]
    fn test_to_samples() {
        let sample_length = 5;
        let values = vec![
            vec![0_u16; sample_length],
            vec![2_u16, 0_u16, 0_u16, 0_u16, 0_u16],
            vec![2_u16; sample_length],
        ];
        let raw_samples = RawSamples {
            values,
            sample_length,
        };
        let samples: Samples = (&raw_samples).into();
        // check the number of samples
        assert_eq!(
            samples.sample_length,
            next_power_of_two(raw_samples.sample_length).unwrap()
        );
        // check length of individual samples
        assert_eq!(
            samples.values.len(),
            next_power_of_two(raw_samples.values.len()).unwrap()
        );
        for sample in &samples.values {
            assert_eq!(sample.len(), samples.sample_length);
        }
    }

    #[test]
    fn test_circuitize_samples() {
        let sample_length = 5;
        let values = vec![
            vec![0_u16; sample_length],
            vec![2_u16, 0_u16, 0_u16, 0_u16, 0_u16],
            vec![2_u16; sample_length],
        ];
        let raw_samples = RawSamples {
            values,
            sample_length,
        };
        let samples: Samples = (&raw_samples).into();
        let csamples: CircuitizedSamples<Fr> = (&samples).into();
        assert_eq!(csamples.len(), samples.values.len());
        csamples
            .iter()
            .for_each(|sample| assert_eq!(sample.len(), samples.sample_length));
    }

    #[test]
    fn test_circuitize_auxiliaries() {
        let sample_length = 5;
        let values = vec![
            vec![0_u16; sample_length],
            vec![2_u16, 0_u16, 0_u16, 0_u16, 0_u16],
            vec![2_u16; sample_length],
        ];
        let raw_samples = RawSamples {
            values,
            sample_length,
        };
        let tree = build_small_tree();
        let raw_trees_model = RawTreesModel {
            trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
            bias: 1.1,
            scale: 6.6,
            n_features: sample_length,
        };
        let trees_model: TreesModel = (&raw_trees_model).into();
        let samples: Samples = (&raw_samples).into();
        let caux = circuitize_auxiliaries::<Fr>(&samples, &trees_model);
        let n_trees = raw_trees_model.trees.len();

        // check dimensions of attributes on paths
        assert_eq!(caux.attributes_on_paths.len(), n_trees);
        for attributes_on_paths_for_tree in &caux.attributes_on_paths {
            assert_eq!(attributes_on_paths_for_tree.len(), samples.values.len());
            for attributes_on_path in attributes_on_paths_for_tree {
                assert_eq!(attributes_on_path.len(), trees_model.depth - 1);
            }
        }
        // check the contents of the attributes on paths for the non-trivial tree (the 0th)
        let attributes_on_path = &caux.attributes_on_paths[0][0];
        assert_eq!(attributes_on_path[0].attr_id, Fr::from(1));
        // ... sample travels left down the tree
        assert_eq!(attributes_on_path[1].attr_id, Fr::from(0));

        // check the dimension of the decision paths
        assert_eq!(caux.decision_paths.len(), n_trees);
        for decision_paths_for_tree in &caux.decision_paths {
            assert_eq!(decision_paths_for_tree.len(), samples.values.len());
            for decision_path in decision_paths_for_tree {
                assert_eq!(decision_path.len(), trees_model.depth - 1);
            }
        }
        // check decision path contents for one combination
        let decision_path = &caux.decision_paths[0][2];
        assert_eq!(decision_path[0].node_id, Fr::from(0));
        // ... sample travels right down the tree
        assert_eq!(decision_path[1].node_id, Fr::from(2));

        // check the dimension of the path_ends
        assert_eq!(caux.path_ends.len(), n_trees);
        for path_ends_for_tree in &caux.path_ends {
            assert_eq!(path_ends_for_tree.len(), samples.values.len());
        }
        // check the contents
        let path_ends_for_tree = &caux.path_ends[0];
        assert_eq!(path_ends_for_tree[0].node_id, Fr::from(3));
        assert_eq!(path_ends_for_tree[1].node_id, Fr::from(4));

        // check the dimensions of the differences
        assert_eq!(caux.differences.len(), n_trees);
        for differences_for_tree in &caux.differences {
            assert_eq!(differences_for_tree.len(), samples.values.len());
            for differences in differences_for_tree {
                assert_eq!(differences.len(), trees_model.depth - 1);
            }
        }
        // check contents
        let differences = &caux.differences[0][0];
        // should have the bit decomposition of -2
        assert_eq!(differences[1].bits[0], Fr::from(0));
        assert_eq!(differences[1].bits[1], Fr::from(1));
        assert_eq!(differences[1].bits[2], Fr::from(0));
        assert_eq!(differences[1].bits[15], Fr::from(1));

        // check the node_multiplicities
        let node_multiplicities = caux.node_multiplicities;
        let n_nodes = 2_usize.pow(trees_model.depth as u32); // includes dummy node
        assert_eq!(node_multiplicities.len(), n_trees);
        for node_multiplicities_for_tree in &node_multiplicities {
            assert_eq!(node_multiplicities_for_tree.len(), n_nodes);
            // root node id has multiplicity samples.values.len() = 4 (not 3, since post padding!)
            let multiplicity = &node_multiplicities_for_tree[0];
            assert_eq!(multiplicity.bits[0], Fr::from(0));
            assert_eq!(multiplicity.bits[1], Fr::from(0));
            assert_eq!(multiplicity.bits[2], Fr::from(1));
            assert_eq!(multiplicity.bits[3], Fr::from(0));
            // dummy node id has multiplicity 0
            // dummy node multiplicity is situated between internal and leaf nodes
            let multiplicity = &node_multiplicities_for_tree[n_nodes / 2 - 1];
            for bit in multiplicity.bits {
                assert_eq!(bit, Fr::from(0));
            }
        }
        // check all node multiplicities for tree 1
        let node_multiplicities_for_tree = &node_multiplicities[0];
        let expected: Vec<usize> = vec![4, 3, 1, 0, 2, 1, 0, 1];
        node_multiplicities_for_tree
            .iter()
            .zip(expected.iter())
            .for_each(|(bits, expected)| {
                let expected_bits = build_node_multiplicity_bindecomp::<Fr>(*expected);
                assert_eq!(expected_bits.bits, bits.bits);
            });

        // FIXME TO BE REMOVED (@Ben)
        // check attribute multiplicities
        // first, check their dimensions
        assert_eq!(caux.attribute_multiplicities.len(), n_trees);
        for am_for_tree in &caux.attribute_multiplicities {
            assert_eq!(am_for_tree.len(), samples.values.len());
            for am_for_tree_and_sample in am_for_tree {
                assert_eq!(am_for_tree_and_sample.len(), samples.sample_length);
            }
        }
        // check some particular values
        let am = &caux.attribute_multiplicities[0][0];
        let mut expected: Vec<usize> = vec![0; samples.sample_length];
        expected[0] = 1;
        expected[1] = 1;
        am.iter().zip(expected.iter()).for_each(|(bits, expected)| {
            let expected_bits = build_4bit_attribute_multiplicity_bindecomp::<Fr>(*expected);
            assert_eq!(expected_bits.bits, bits.bits);
        })
    }

    #[test]
    fn test_attribute_multiplicities_per_sample() {
        let sample_length = 5;
        let values = vec![
            vec![0_u16; sample_length],
            vec![2_u16, 0_u16, 0_u16, 0_u16, 0_u16],
            vec![2_u16; sample_length],
        ];
        let raw_samples = RawSamples {
            values,
            sample_length,
        };

        let raw_trees_model = RawTreesModel {
            trees: vec![build_small_tree(), build_small_tree_variant()],
            bias: 1.1,
            scale: 6.6,
            n_features: sample_length,
        };
        let trees_model: TreesModel = (&raw_trees_model).into();
        let samples: Samples = (&raw_samples).into();
        let caux = circuitize_auxiliaries::<Fr>(&samples, &trees_model);

        let am = caux.attribute_multiplicities_per_sample;
        // check dimensions
        assert_eq!(am.len(), samples.values.len());
        for am_for_sample in &am {
            assert_eq!(am_for_sample.len(), samples.sample_length);
        }
        // check some values
        assert_eq!(
            am[0][0],
            build_8bit_attribute_multiplicity_bindecomp::<Fr>(1)
        );
        assert_eq!(
            am[0][1],
            build_8bit_attribute_multiplicity_bindecomp::<Fr>(1)
        );
        assert_eq!(
            am[0][2],
            build_8bit_attribute_multiplicity_bindecomp::<Fr>(0)
        );
        assert_eq!(
            am[0][3],
            build_8bit_attribute_multiplicity_bindecomp::<Fr>(1)
        );
        assert_eq!(
            am[0][4],
            build_8bit_attribute_multiplicity_bindecomp::<Fr>(1)
        );
        assert_eq!(
            am[0][5],
            build_8bit_attribute_multiplicity_bindecomp::<Fr>(0)
        );
        assert_eq!(
            am[2][0],
            build_8bit_attribute_multiplicity_bindecomp::<Fr>(2)
        );
        assert_eq!(
            am[2][1],
            build_8bit_attribute_multiplicity_bindecomp::<Fr>(1)
        );
        assert_eq!(
            am[2][2],
            build_8bit_attribute_multiplicity_bindecomp::<Fr>(0)
        );
        assert_eq!(
            am[2][3],
            build_8bit_attribute_multiplicity_bindecomp::<Fr>(1)
        );
        assert_eq!(
            am[2][4],
            build_8bit_attribute_multiplicity_bindecomp::<Fr>(0)
        );
        assert_eq!(
            am[2][5],
            build_8bit_attribute_multiplicity_bindecomp::<Fr>(0)
        );
    }

    #[test]
    fn test_trees_model_from() {
        let tree = build_small_tree();
        let raw_trees_model = RawTreesModel {
            trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
            bias: 1.1,
            scale: 6.6,
            n_features: 11,
        };
        let trees_model: TreesModel = (&raw_trees_model).into();
        assert_eq!(trees_model.trees.len(), 2);
        assert_eq!(trees_model.depth, 3);
        // check trees are as claimed
        for tree in &trees_model.trees {
            assert_eq!(tree.depth(std::cmp::max), 3);
            assert!(tree.is_perfect());
            assert_eq!(tree.get_id().unwrap(), 0);
        }
    }

    #[test]
    fn test_circuitized_trees_from() {
        let tree = build_small_tree();
        let raw_trees_model = RawTreesModel {
            trees: vec![tree, Node::new_leaf(Some(0), 3.0)],
            bias: 1.1,
            scale: 6.6,
            n_features: 11,
        };
        let trees_model: TreesModel = (&raw_trees_model).into();
        let ctrees: CircuitizedTrees<Fr> = (&trees_model).into();
        assert_eq!(ctrees.depth, trees_model.depth);
        // check decision nodes
        for tree_dns in &ctrees.decision_nodes {
            assert_eq!(tree_dns.len(), 4);
            for (node_id, node) in tree_dns.iter().take(tree_dns.len() - 1).enumerate() {
                assert_eq!(node.node_id, Fr::from(node_id as u64));
            }
            assert_eq!(tree_dns[tree_dns.len() - 1].node_id, Fr::from(7_u64));
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
        let expected_score = raw_trees_model.scale * (0.1 + 3.0) + raw_trees_model.bias;
        let quant_score = (expected_score * ctrees.scaling) as i64;
        let f_quant_score = if quant_score >= 0 {
            Fr::from(quant_score as u64)
        } else {
            -Fr::from(quant_score.unsigned_abs())
        };
        // just check that's it's close
        assert_eq!(f_quant_score, acc_score);
    }

    #[test]
    fn test_generate_raw_trees_model() {
        let target_depth = 3;
        let n_features = 6;
        let n_trees = 11;
        let raw_trees_model = generate_raw_trees_model(n_trees, target_depth, n_features, 0.5);
        assert_eq!(raw_trees_model.n_features, n_features);
        assert_eq!(raw_trees_model.trees.len(), n_trees);
        for tree in &raw_trees_model.trees {
            assert!(tree.depth(std::cmp::max) <= target_depth);
        }
    }

    #[test]
    fn test_generate_raw_samples() {
        let raw_samples = generate_raw_samples(10, 3);
        assert_eq!(raw_samples.values.len(), 10);
        for row in &raw_samples.values {
            assert_eq!(row.len(), 3);
        }
    }

    #[test]
    fn test_documentation() {
        // TODO remove once the doctests are being run
        let n_trees = 3;
        let n_features = 4;
        let depth = 6;
        // start with some random (probably not perfect) trees with f64 leaf values
        let raw_trees_model = generate_raw_trees_model(n_trees, depth, n_features, 0.5);
        // pad the trees, assign ids and quantize the leaf values
        let trees_model: TreesModel = (&raw_trees_model).into();
        // circuitize the trees (converts to DecisionNode<F>, LeafNode<F>)
        let _ctrees: CircuitizedTrees<Fr> = (&trees_model).into();
        // .. continued
        // generate some samples to play with
        let n_samples = 10;
        let raw_samples = generate_raw_samples(n_samples, n_features);
        let samples: Samples = (&raw_samples).into();
        let _csamples: CircuitizedSamples<Fr> = (&samples).into();
        // notice: circuitize_auxiliaries takes trees_model, not ctrees!
        let _caux = circuitize_auxiliaries::<Fr>(&samples, &trees_model);
        // .. continued
        let _raw_trees_model: RawTreesModel =
            load_raw_trees_model(Path::new("src/zkdt/data_pipeline/test_qtrees.json"));
        let _raw_samples: RawSamples =
            load_raw_samples(Path::new("src/zkdt/data_pipeline/test_samples_10x6.npy"));
    }

    #[test]
    fn test_upshot_loading_and_circuitization() {
        // for this to work, those files need to be in place (not stored on the repo):
        // 1. remainder_prover/upshot_data/upshot-quantized-samples.npy
        // 2. remainder_prover/upshot_data/quantized-upshot-model.json
        let raw_trees_model: RawTreesModel =
            load_raw_trees_model(Path::new("upshot_data/quantized-upshot-model.json"));
        let mut raw_samples: RawSamples =
            load_raw_samples(Path::new("upshot_data/upshot-quantized-samples.npy"));
        // use just a small batch
        raw_samples.values = raw_samples.values[0..4].to_vec();

        let mut trees_model: TreesModel = (&raw_trees_model).into();
        // just take two trees
        trees_model = trees_model.slice(0..2);

        let samples: Samples = (&raw_samples).into();

        let _ctrees: CircuitizedTrees<Fr> = (&trees_model).into();
        let _csamples: CircuitizedSamples<Fr> = (&samples).into();
        let _caux = circuitize_auxiliaries::<Fr>(&samples, &trees_model);
    }

    #[test]
    fn test_count_node_multiplicities_null() {
        let tree_depth = 3;
        let tree_paths: Vec<TreePath<i32>> = vec![];
        // test first for no tree paths at all
        let multiplicities = count_node_multiplicities(&tree_paths, tree_depth);
        assert_eq!(multiplicities.len(), 2_usize.pow(tree_depth as u32));
        multiplicities.iter().for_each(|mult| assert_eq!(*mult, 0));
    }

    #[test]
    fn test_count_node_multiplicities_nontrivial() {
        let depth = 3;
        let mut tree = build_small_tree().perfect_to_depth(depth);
        tree.assign_id(0);
        let sample_length = 5;
        let values = vec![
            vec![0_u16; sample_length],
            vec![2_u16, 0_u16, 0_u16, 0_u16, 0_u16],
            vec![2_u16; sample_length],
        ];
        let mut tree_paths: Vec<TreePath<f64>> = vec![];
        for sample in values {
            tree_paths.push(tree.get_tree_path(&sample));
        }
        let multiplicities = count_node_multiplicities(&tree_paths, depth);
        let expected: Vec<usize> = vec![3, 2, 1, 0, 1, 1, 0, 1];
        assert_eq!(multiplicities, expected);
    }

    // FIXME TO BE REMOVED (@Ben)
    #[test]
    fn test_count_attribute_multiplicities() {
        let sample_length: usize = 10;
        let multiplicities = count_attribute_multiplicities(&vec![], sample_length);
        assert_eq!(multiplicities.len(), 10);
        assert_eq!(multiplicities[1], 0);
        let mut path: Vec<PathStep> = vec![];
        for i in 0..2 {
            path.push(PathStep {
                node_id: 1,
                feature_index: 2,
                threshold: 3,
                feature_value: 4,
            });
        }
        let multiplicities = count_attribute_multiplicities(&path, sample_length);
        assert_eq!(multiplicities.len(), 10);
        assert_eq!(multiplicities[1], 0);
        assert_eq!(multiplicities[2], 2);
        assert_eq!(multiplicities[3], 0);
        assert_eq!(multiplicities[4], 0);
    }

    #[test]
    fn test_build_sample_witness() {
        let sample: Vec<u16> = vec![3, 1, 4];
        let witness = build_sample_witness::<Fr>(&sample);
        assert_eq!(witness.len(), sample.len());
    }

    #[test]
    fn test_pad_tree_count_to_multiple() {
        let n_trees = 7;
        let n_features = 4;
        let depth = 6;
        // start with some random (probably not perfect) trees with f64 leaf values
        let raw_trees_model = generate_raw_trees_model(n_trees, depth, n_features, 0.5);
        // padding to multiple of 1 should be noop
        let mut padded_trees_model: TreesModel = (&raw_trees_model).into();
        padded_trees_model.pad_tree_count_to_multiple(1);
        assert_eq!(padded_trees_model.trees.len(), n_trees);
        // padding with multiple equal to the tree count should be a noop
        let mut padded_trees_model: TreesModel = (&raw_trees_model).into();
        padded_trees_model.pad_tree_count_to_multiple(n_trees);
        assert_eq!(padded_trees_model.trees.len(), n_trees);
        // try some non-trival cases
        let mut padded_trees_model: TreesModel = (&raw_trees_model).into();
        padded_trees_model.pad_tree_count_to_multiple(4);
        assert_eq!(padded_trees_model.trees.len(), 8);
        let mut padded_trees_model: TreesModel = (&raw_trees_model).into();
        padded_trees_model.pad_tree_count_to_multiple(15);
        assert_eq!(padded_trees_model.trees.len(), 15);
        // examine some resultant trees by comparing to unpadded
        let trees_model: TreesModel = (&raw_trees_model).into();
        let mut padded_trees_model: TreesModel = (&raw_trees_model).into();
        padded_trees_model.pad_tree_count_to_multiple(5);
        assert_eq!(padded_trees_model.trees.len(), 10);
        // check that the first 7 trees are the same as the original trees
        trees_model
            .trees
            .iter()
            .zip(padded_trees_model.trees.iter())
            .for_each(|(tree, padded_tree)| {
                assert_eq!(tree.depth(std::cmp::max), padded_tree.depth(std::cmp::max));
                assert_eq!(tree.get_id(), padded_tree.get_id());
                if let Node::Internal {
                    feature_index,
                    threshold,
                    ..
                } = tree
                {
                    if let Node::Internal {
                        feature_index: padded_feature_index,
                        threshold: padded_threshold,
                        ..
                    } = padded_tree
                    {
                        assert_eq!(feature_index, padded_feature_index);
                        assert_eq!(threshold, padded_threshold);
                    } else {
                        panic!("Expected internal node, got leaf node.");
                    }
                } else {
                    panic!("Expected internal node, got leaf node.");
                }
                assert!(tree.is_perfect());
                assert!(padded_tree.is_perfect());
            });
        // check that the last 3 trees are perfect
        padded_trees_model.trees[7..10].iter().for_each(|tree| {
            assert_eq!(tree.depth(std::cmp::max), trees_model.depth);
            assert!(tree.is_perfect());
        });
        // get a TreePath, and check that the leaf value is 0.0
        let sample = vec![0_u16; n_features];
        let tree_path = padded_trees_model.trees[7].get_tree_path(&sample);
        assert_eq!(tree_path.leaf_value, 0_i64);
    }
}
