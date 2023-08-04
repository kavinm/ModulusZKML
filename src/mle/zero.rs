use itertools::{repeat_n, Itertools};

use crate::{
    layer::{Claim, LayerId},
    FieldExt,
};

use super::{MleIndex, MleRef};

///An MLERef that is only zeros; Typically used for the output layer
#[derive(Debug)]
pub struct ZeroMleRef<F: FieldExt> {
    mle_indices: Vec<MleIndex<F>>,
    /// Number of non-fixed variables within this MLE
    /// (warning: this gets modified destructively DURING sumcheck)
    num_vars: usize,
    layer_id: Option<LayerId>,
    zero: [F; 1],
    indexed: bool,
}

impl<F: FieldExt> ZeroMleRef<F> {
    ///Creates a new ZERO MleRef
    pub fn new(num_vars: usize, prefix_bits: Option<Vec<MleIndex<F>>>, layer_id: LayerId) -> Self {
        let mle_indices = prefix_bits
            .into_iter()
            .flatten()
            .chain(repeat_n(MleIndex::Iterated, num_vars))
            .collect_vec();

        Self {
            mle_indices,
            num_vars,
            layer_id: Some(layer_id),
            zero: [F::zero()],
            indexed: false,
        }
    }
}

impl<F: FieldExt> MleRef for ZeroMleRef<F> {
    type F = F;

    fn bookkeeping_table(&self) -> &[Self::F] {
        &self.zero
    }

    fn indexed(&self) -> bool {
        self.indexed
    }

    fn mle_indices(&self) -> &[MleIndex<Self::F>] {
        &self.mle_indices
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn fix_variable(&mut self, round_index: usize, challenge: Self::F) -> Option<Claim<Self::F>> {
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::IndexedBit(round_index) {
                *mle_index = MleIndex::Bound(challenge);
            }
        }

        // --- One fewer iterated bit to sumcheck through ---
        self.num_vars -= 1;

        if self.num_vars == 0 {
            Some((
                self.mle_indices
                    .iter()
                    .map(|x| match x {
                        MleIndex::Bound(chal) => *chal,
                        MleIndex::Fixed(bit) => {
                            if *bit {
                                F::one()
                            } else {
                                F::zero()
                            }
                        }
                        _ => panic!("All bits should be bound!"),
                    })
                    .collect_vec(),
                F::zero(),
            ))
        } else {
            None
        }
    }

    fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        let mut new_indices = 0;
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::Iterated {
                *mle_index = MleIndex::IndexedBit(curr_index + new_indices);
                new_indices += 1;
            }
        }

        curr_index + new_indices
    }

    fn get_layer_id(&self) -> Option<LayerId> {
        self.layer_id.clone()
    }
}
