//! Functions for loading and converting ZKDT data into a form-factor
//! ready for input into the ZKDT circuit

use std::path::Path;

use ark_std::log2;
use itertools::Itertools;
use remainder_shared_types::FieldExt;
use serde::{Serialize, Deserialize};
use tracing::instrument;

use crate::{mle::dense::DenseMle, layer::LayerId, prover::input_layer};
use super::{data_pipeline::dt2zkdt::{load_raw_trees_model, RawTreesModel, load_raw_samples, RawSamples, TreesModel, Samples, CircuitizedTrees, CircuitizedSamples, CircuitizedAuxiliaries, circuitize_auxiliaries}, structs::{InputAttribute, DecisionNode, BinDecomp16Bit, LeafNode, BinDecomp4Bit, BinDecomp8Bit}};

#[derive(Clone)]
pub struct BatchedZKDTCircuitMles<F: FieldExt> {
    pub input_samples_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
    pub permuted_input_samples_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
    pub decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>,
    pub leaf_node_paths_mle_vec: Vec<DenseMle<F, LeafNode<F>>>,
    pub binary_decomp_diffs_mle_vec: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    pub multiplicities_bin_decomp_mle_decision: DenseMle<F, BinDecomp16Bit<F>>,
    pub multiplicities_bin_decomp_mle_leaf: DenseMle<F, BinDecomp16Bit<F>>,
    pub decision_nodes_mle: DenseMle<F, DecisionNode<F>>,
    pub leaf_nodes_mle: DenseMle<F, LeafNode<F>>,
    pub multiplicities_bin_decomp_mle_input_vec: Vec<DenseMle<F, BinDecomp8Bit<F>>>,
}

#[derive(Clone)]
pub struct BatchedZKDTCircuitMlesMultiTree<F: FieldExt> {
    pub input_samples_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
    pub permuted_input_samples_mle_vec_vec: Vec<Vec<DenseMle<F, InputAttribute<F>>>>,
    pub decision_node_paths_mle_vec_vec: Vec<Vec<DenseMle<F, DecisionNode<F>>>>,
    pub leaf_node_paths_mle_vec_vec: Vec<Vec<DenseMle<F, LeafNode<F>>>>,
    pub binary_decomp_diffs_mle_vec_vec: Vec<Vec<DenseMle<F, BinDecomp16Bit<F>>>>,
    pub multiplicities_bin_decomp_mle_decision_vec: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    pub multiplicities_bin_decomp_mle_leaf_vec: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    pub decision_nodes_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>,
    pub leaf_nodes_mle_vec: Vec<DenseMle<F, LeafNode<F>>>,
    pub multiplicities_bin_decomp_mle_input: Vec<DenseMle<F, BinDecomp8Bit<F>>>,
}

#[derive(Serialize, Deserialize)]
pub struct ZKDTCircuitData<F> {
    pub input_data: Vec<Vec<InputAttribute<F>>>, // Input attributes
    pub permuted_input_data: Vec<Vec<InputAttribute<F>>>, // Permuted input attributes
    pub decision_node_paths: Vec<Vec<DecisionNode<F>>>, // Paths (decision node part only)
    pub leaf_node_paths: Vec<LeafNode<F>>,       // Paths (leaf node part only)
    pub binary_decomp_diffs: Vec<Vec<BinDecomp16Bit<F>>>, // Binary decomp of differences
    pub multiplicities_bin_decomp: Vec<BinDecomp16Bit<F>>, // Binary decomp of multiplicities
    pub decision_nodes: Vec<DecisionNode<F>>,    // Actual tree decision nodes
    pub leaf_nodes: Vec<LeafNode<F>>,            // Actual tree leaf nodes
    pub multiplicities_bin_decomp_input: Vec<Vec<BinDecomp8Bit<F>>>, // Binary decomp of multiplicities, of input
}

impl<F: FieldExt> ZKDTCircuitData<F> {
    /// Constructor
    pub fn new(
        input_data: Vec<Vec<InputAttribute<F>>>,
        permuted_input_data: Vec<Vec<InputAttribute<F>>>,
        decision_node_paths: Vec<Vec<DecisionNode<F>>>,
        leaf_node_paths: Vec<LeafNode<F>>,
        binary_decomp_diffs: Vec<Vec<BinDecomp16Bit<F>>>,
        multiplicities_bin_decomp: Vec<BinDecomp16Bit<F>>,
        decision_nodes: Vec<DecisionNode<F>>,
        leaf_nodes: Vec<LeafNode<F>>,
        multiplicities_bin_decomp_input: Vec<Vec<BinDecomp8Bit<F>>>,
    ) -> ZKDTCircuitData<F> {
        ZKDTCircuitData {
            input_data,
            permuted_input_data,
            decision_node_paths,
            leaf_node_paths,
            binary_decomp_diffs,
            multiplicities_bin_decomp,
            decision_nodes,
            leaf_nodes,
            multiplicities_bin_decomp_input
        }
    }
}


#[instrument(skip(zkdt_circuit_data))]
pub fn convert_zkdt_circuit_data_multi_tree_into_mles<F: FieldExt>(
    zkdt_circuit_data: Vec<ZKDTCircuitData<F>>,
    tree_height: usize,
    input_len: usize,
) -> (BatchedZKDTCircuitMlesMultiTree<F>, (usize, usize)) {

    let (
        input_data_vec,
        permuted_input_data_vec,
        decision_node_paths_vec,
        leaf_node_paths_vec,
        binary_decomp_diffs_vec,
        mut multiplicities_bin_decomp_vec,
        decision_nodes_vec,
        leaf_nodes_vec,
        multiplicities_bin_decomp_input_vec
    ): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>,) = zkdt_circuit_data.into_iter().map(
        |zkdt_circuit_data| {
            (
                zkdt_circuit_data.input_data,
                zkdt_circuit_data.permuted_input_data,
                zkdt_circuit_data.decision_node_paths,
                zkdt_circuit_data.leaf_node_paths,
                zkdt_circuit_data.binary_decomp_diffs,
                zkdt_circuit_data.multiplicities_bin_decomp,
                zkdt_circuit_data.decision_nodes,
                zkdt_circuit_data.leaf_nodes,
                zkdt_circuit_data.multiplicities_bin_decomp_input,
            )
        }
    ).multiunzip();

    let decision_len = 2_usize.pow(tree_height as u32 - 1);

    // --- Generate MLEs for each ---
    let (multiplicities_bin_decomp_mle_decision_vec, multiplicities_bin_decomp_mle_leaf_vec): (Vec<_>, Vec<_>) = multiplicities_bin_decomp_vec.into_iter().map(
        |mut multiplicities_bin_decomp| {
            let multiplicities_bin_decomp_leaf = multiplicities_bin_decomp.split_off(decision_len);
            let multiplicities_bin_decomp_decision = multiplicities_bin_decomp;
            let multiplicities_bin_decomp_mle_decision = DenseMle::new_from_iter(multiplicities_bin_decomp_decision
                .into_iter()
                .map(BinDecomp16Bit::from), LayerId::Input(0), None);
            let multiplicities_bin_decomp_mle_leaf = DenseMle::new_from_iter(multiplicities_bin_decomp_leaf
                .into_iter()
                .map(BinDecomp16Bit::from), LayerId::Input(0), None);
            (multiplicities_bin_decomp_mle_decision, multiplicities_bin_decomp_mle_leaf)
        }
    ).unzip();

    let input_samples_mle_vec = 
        input_data_vec[0].clone().into_iter().map(|input| DenseMle::new_from_iter(input
                .into_iter()
                .map(InputAttribute::from), LayerId::Input(0), None)).collect_vec();

    let permuted_input_samples_mle_vec_vec = permuted_input_data_vec.into_iter().map(
        |permuted_input_data| {
            permuted_input_data
            .iter().map(|datum| DenseMle::new_from_iter(datum
                .clone()
                .into_iter()
                .map(InputAttribute::from), LayerId::Input(0), None)).collect()
        }
    ).collect_vec();


    let decision_node_paths_mle_vec_vec: Vec<Vec<DenseMle<F, DecisionNode<F>>>> = decision_node_paths_vec.into_iter().map(
        |decision_node_paths| {
            decision_node_paths
        .iter()
        .map(|path|
            DenseMle::new_from_iter(path
            .clone()
            .into_iter(), LayerId::Input(0), None))
        .collect()
            }).collect_vec();

    let leaf_node_paths_mle_vec_vec = leaf_node_paths_vec.into_iter().map(
        |leaf_node_paths| {
            leaf_node_paths.into_iter()
            .map(|path| DenseMle::new_from_iter([path].into_iter(), LayerId::Input(0), None))
            .collect()
        }
    ).collect_vec();

    let binary_decomp_diffs_mle_vec_vec = binary_decomp_diffs_vec.into_iter().map(
        |binary_decomp_diffs| {
        binary_decomp_diffs.iter()
        .map(|binary_decomp_diff|
            DenseMle::new_from_iter(binary_decomp_diff
                .clone()
                .into_iter()
                .map(BinDecomp16Bit::from), LayerId::Input(0), None))
        .collect_vec()
            }).collect_vec();

    let decision_nodes_mle_vec = decision_nodes_vec.into_iter().map(
        |decision_nodes| {
            DenseMle::new_from_iter(decision_nodes
                .into_iter()
                .map(DecisionNode::from), LayerId::Input(0), None)
        }
    ).collect_vec();

    let leaf_nodes_mle_vec = leaf_nodes_vec.into_iter().map(
        |leaf_nodes| {
            DenseMle::new_from_iter(leaf_nodes
                .into_iter()
                .map(LeafNode::from), LayerId::Input(0), None)
        }
    ).collect_vec();

    let multiplicities_bin_decomp_mle_input = multiplicities_bin_decomp_input_vec[0]
        .iter().map(|datum|
            DenseMle::new_from_iter(datum
            .clone()
            .into_iter()
            .map(BinDecomp8Bit::from), LayerId::Input(0), None))
        .collect_vec();


    (BatchedZKDTCircuitMlesMultiTree {
        input_samples_mle_vec,
        permuted_input_samples_mle_vec_vec,
        decision_node_paths_mle_vec_vec,
        leaf_node_paths_mle_vec_vec,
        binary_decomp_diffs_mle_vec_vec,
        multiplicities_bin_decomp_mle_decision_vec,
        multiplicities_bin_decomp_mle_leaf_vec,
        decision_nodes_mle_vec,
        leaf_nodes_mle_vec,
        multiplicities_bin_decomp_mle_input,
    }, (tree_height, input_len))
}

/// Takes the output from presumably something like [`read_upshot_data_single_tree_branch_from_filepath`]
/// and converts it into `BatchedCatboostMles<F>`, i.e. the input to the circuit.
#[instrument(skip(zkdt_circuit_data))]
pub fn convert_zkdt_circuit_data_into_mles<F: FieldExt>(
    zkdt_circuit_data: ZKDTCircuitData<F>,
    tree_height: usize,
    input_len: usize,
) -> (BatchedZKDTCircuitMles<F>, (usize, usize)) {

    // --- Unpacking ---
    let ZKDTCircuitData {
        input_data,
        permuted_input_data,
        decision_node_paths,
        leaf_node_paths,
        binary_decomp_diffs,
        mut multiplicities_bin_decomp,
        decision_nodes,
        leaf_nodes,
        multiplicities_bin_decomp_input,
    } = zkdt_circuit_data;

    let decision_len = 2_usize.pow(tree_height as u32 - 1);
    let multiplicities_bin_decomp_leaf = multiplicities_bin_decomp.split_off(decision_len);
    let multiplicities_bin_decomp_decision = multiplicities_bin_decomp;

    // --- Generate MLEs for each ---
    let input_samples_mle_vec = input_data.into_iter().map(|input| DenseMle::new_from_iter(input
        .into_iter()
        .map(InputAttribute::from), LayerId::Input(0), None)).collect_vec();
    let permuted_input_samples_mle_vec = permuted_input_data
        .iter().map(|datum| DenseMle::new_from_iter(datum
            .clone()
            .into_iter()
            .map(InputAttribute::from), LayerId::Input(0), None)).collect();
    let decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>> = decision_node_paths
        .iter()
        .map(|path|
            DenseMle::new_from_iter(path
            .clone()
            .into_iter(), LayerId::Input(0), None))
        .collect();
    let leaf_node_paths_mle_vec = leaf_node_paths
        .into_iter()
        .map(|path| DenseMle::new_from_iter([path].into_iter(), LayerId::Input(0), None))
        .collect();
    let binary_decomp_diffs_mle_vec = binary_decomp_diffs
        .iter()
        .map(|binary_decomp_diff|
            DenseMle::new_from_iter(binary_decomp_diff
                .clone()
                .into_iter()
                .map(BinDecomp16Bit::from), LayerId::Input(0), None))
        .collect_vec();
    let multiplicities_bin_decomp_mle_decision = DenseMle::new_from_iter(multiplicities_bin_decomp_decision
        .into_iter()
        .map(BinDecomp16Bit::from), LayerId::Input(0), None);
    let multiplicities_bin_decomp_mle_leaf = DenseMle::new_from_iter(multiplicities_bin_decomp_leaf
        .into_iter()
        .map(BinDecomp16Bit::from), LayerId::Input(0), None);
    let decision_nodes_mle = DenseMle::new_from_iter(decision_nodes
        .into_iter()
        .map(DecisionNode::from), LayerId::Input(0), None);
    let leaf_nodes_mle = DenseMle::new_from_iter(leaf_nodes
        .into_iter()
        .map(LeafNode::from), LayerId::Input(0), None);
    let multiplicities_bin_decomp_mle_input = multiplicities_bin_decomp_input
        .iter().map(|datum|
            DenseMle::new_from_iter(datum
            .clone()
            .into_iter()
            .map(BinDecomp8Bit::from), LayerId::Input(0), None))
        .collect_vec();

    (BatchedZKDTCircuitMles {
        input_samples_mle_vec,
        permuted_input_samples_mle_vec,
        decision_node_paths_mle_vec,
        leaf_node_paths_mle_vec,
        binary_decomp_diffs_mle_vec,
        multiplicities_bin_decomp_mle_decision,
        multiplicities_bin_decomp_mle_leaf,
        decision_nodes_mle,
        leaf_nodes_mle,
        multiplicities_bin_decomp_mle_input_vec: multiplicities_bin_decomp_mle_input
    }, (tree_height, input_len))
}

/// Specifies exactly which minibatch to use within a sample.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MinibatchData {
    /// log_2 of the minibatch size
    pub log_sample_minibatch_size: usize,
    /// Minibatch index within the bigger batch
    pub sample_minibatch_number: usize,
}

/// Gives all batched data associated with tree number `tree_idx`.
/// 
/// ## Arguments
/// * `maybe_minibatch_data` - The minibatch to grab data for, including minibatch size and index.
/// * `tree_idx` - The tree number we are generating witnesses for.
/// * `raw_trees_model_path` - Path to the JSON file representing the quantized version of the model
///     (as output by the Python preprocessing)
/// * `raw_samples_path` - Path to the NumPy file representing the quantized version of the samples
///     (again, as output by the Python preprocessing)
/// 
/// ## Returns
/// * `zkdt_circuit_data` - Data which is ready to be thrown into circuit
/// * `tree_height` - Length of every path within the given tree
/// * `input_len` - Padded number of features within each input
/// * `log_minibatch_size` - log_2 of the size of the minibatch, i.e. number of inputs being loaded.
/// * `minibatch_number` - Minibatch index we are processing
/// 
/// ## Notes
/// Note that `raw_samples.values.len()` is currently 4573! This means we can go
/// up to 8192 (padded) in terms of batch sizes which are powers of 2
/// 
/// ## TODOs
/// * Throw an error if `sample_minibatch_number` causes us to go out of bounds!
#[instrument]
pub fn load_upshot_data_single_tree_batch<F: FieldExt>(
    maybe_minibatch_data: Option<MinibatchData>,
    tree_idx: usize,
    raw_trees_model_path: &Path,
    raw_samples_path: &Path,
) -> (ZKDTCircuitData<F>, (usize, usize), MinibatchData) {

    // --- Grab trees + raw samples ---
    let raw_trees_model: RawTreesModel = load_raw_trees_model(raw_trees_model_path);
    let mut raw_samples: RawSamples = load_raw_samples(raw_samples_path);

    // --- Grab sample minibatch ---
    let minibatch_data = match maybe_minibatch_data {
        Some(param_minibatch_data) => param_minibatch_data,
        None => {
            MinibatchData {
                sample_minibatch_number: 0,
                log_sample_minibatch_size: log2(raw_samples.values.len() as usize) as usize,
            }
        }
    };
    let sample_minibatch_size = 2_usize.pow(minibatch_data.log_sample_minibatch_size as u32);
    let minibatch_start_idx = minibatch_data.sample_minibatch_number * sample_minibatch_size;
    raw_samples.values = raw_samples.values[minibatch_start_idx..(minibatch_start_idx + sample_minibatch_size)].to_vec();

    // --- Conversions ---
    let full_trees_model: TreesModel = (&raw_trees_model).into();
    let single_tree = full_trees_model.slice(tree_idx..tree_idx + 1);
    let samples: Samples = (&raw_samples).into();
    let ctrees: CircuitizedTrees<F> = (&single_tree).into();

    // --- Compute actual witnesses ---
    let csamples: CircuitizedSamples<F> = (&samples).into();
    let caux = circuitize_auxiliaries(&samples, &single_tree);
    let tree_height = ctrees.depth;
    let input_len = csamples[0].len();

    // --- Sanitycheck ---
    debug_assert_eq!(caux.attributes_on_paths.len(), 1);
    debug_assert_eq!(caux.decision_paths.len(), 1);
    debug_assert_eq!(caux.path_ends.len(), 1);
    debug_assert_eq!(caux.differences.len(), 1);
    debug_assert_eq!(caux.node_multiplicities.len(), 1);
    debug_assert_eq!(ctrees.decision_nodes.len(), 1);
    debug_assert_eq!(ctrees.leaf_nodes.len(), 1);
    debug_assert_eq!(caux.attribute_multiplicities.len(), 1);

    // --- Grab only the slice of witnesses which are relevant to the target `tree_number` ---
    (ZKDTCircuitData::new(
        csamples,
        caux.attributes_on_paths[0].clone(),
        caux.decision_paths[0].clone(),
        caux.path_ends[0].clone(),
        caux.differences[0].clone(),
        caux.node_multiplicities[0].clone(),
        ctrees.decision_nodes[0].clone(),
        ctrees.leaf_nodes[0].clone(),
        caux.attribute_multiplicities_per_sample.clone()
    ), (tree_height, input_len), minibatch_data)
}


#[instrument]
pub fn load_upshot_data_multi_tree_batch<F: FieldExt>(
    maybe_minibatch_data: Option<MinibatchData>,
    tree_batch_size: usize,
    tree_batch_number: usize,
    raw_trees_model_path: &Path,
    raw_samples_path: &Path,
) -> (Vec<ZKDTCircuitData<F>>, (usize, usize), MinibatchData) {

    // --- Grab trees + raw samples ---
    let raw_trees_model: RawTreesModel = load_raw_trees_model(raw_trees_model_path);
    let mut raw_samples: RawSamples = load_raw_samples(raw_samples_path);

    // --- Grab sample minibatch ---
    let minibatch_data = match maybe_minibatch_data {
        Some(param_minibatch_data) => param_minibatch_data,
        None => {
            MinibatchData {
                sample_minibatch_number: 0,
                log_sample_minibatch_size: log2(raw_samples.values.len() as usize) as usize,
            }
        }
    };
    let sample_minibatch_size = 2_usize.pow(minibatch_data.log_sample_minibatch_size as u32);
    let minibatch_start_idx = minibatch_data.sample_minibatch_number * sample_minibatch_size;
    raw_samples.values = raw_samples.values[minibatch_start_idx..(minibatch_start_idx + sample_minibatch_size)].to_vec();

    // --- Conversions ---
    let full_trees_model: TreesModel = (&raw_trees_model).into();
    let tree_batch = full_trees_model.slice((tree_batch_size * tree_batch_number)..(tree_batch_size * (tree_batch_number + 1)));
    let samples: Samples = (&raw_samples).into();
    let ctrees: CircuitizedTrees<F> = (&tree_batch).into();

    // --- Compute actual witnesses ---
    let csamples: CircuitizedSamples<F> = (&samples).into();
    let caux = circuitize_auxiliaries(&samples, &tree_batch);
    let tree_height = ctrees.depth;
    let input_len = csamples[0].len();

    // --- Sanitycheck ---
    debug_assert_eq!(caux.attributes_on_paths.len(), tree_batch_size);
    debug_assert_eq!(caux.decision_paths.len(), tree_batch_size);
    debug_assert_eq!(caux.path_ends.len(), tree_batch_size);
    debug_assert_eq!(caux.differences.len(), tree_batch_size);
    debug_assert_eq!(caux.node_multiplicities.len(), tree_batch_size);
    debug_assert_eq!(ctrees.decision_nodes.len(), tree_batch_size);
    debug_assert_eq!(ctrees.leaf_nodes.len(), tree_batch_size);
    // debug_assert_eq!(caux.attribute_multiplicities.len(), tree_batch_number);

    // --- Grab only the slice of witnesses which are relevant to the target `tree_number` ---

    let circuitdata_vec = caux.attributes_on_paths.clone().into_iter().enumerate().map(
        |(idx, _)| {
            ZKDTCircuitData::new(
                csamples.clone(),
                caux.attributes_on_paths[idx].clone(),
                caux.decision_paths[idx].clone(),
                caux.path_ends[idx].clone(),
                caux.differences[idx].clone(),
                caux.node_multiplicities[idx].clone(),
                ctrees.decision_nodes[idx].clone(),
                ctrees.leaf_nodes[idx].clone(),
                caux.attribute_multiplicities_per_sample.clone()
            )
        }
    ).collect_vec();
    (circuitdata_vec, (tree_height, input_len), minibatch_data)
}