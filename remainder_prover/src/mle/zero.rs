// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//!Contains ZeroMleRef which is an MleRef which always contains only all zeros

//!Contains ZeroMleRef which is an MleRef which always contains only all zeros

use itertools::{repeat_n, Itertools};
use serde::{Deserialize, Serialize};

use crate::layer::{claims::Claim, LayerId};
use remainder_shared_types::FieldExt;

use super::{mle_enum::MleEnum, MleIndex, MleRef};

///An MLERef that is only zeros; Typically used for the output layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroMleRef<F> {
    pub(crate) mle_indices: Vec<MleIndex<F>>,
    pub(crate) original_mle_indices: Vec<MleIndex<F>>,
    /// Number of non-fixed variables within this MLE
    /// (warning: this gets modified destructively DURING sumcheck)
    num_vars: usize,
    pub(crate) layer_id: LayerId,
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
            mle_indices: mle_indices.clone(),
            original_mle_indices: mle_indices,
            num_vars,
            layer_id,
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

    fn original_mle_indices(&self) -> &Vec<MleIndex<Self::F>> {
        &self.original_mle_indices
    }

    fn original_bookkeeping_table(&self) -> &Vec<Self::F> {
        &self.original_bookkeeping_table()
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn original_num_vars(&self) -> usize {
        self.num_vars
    }

    fn fix_variable(&mut self, round_index: usize, challenge: Self::F) -> Option<Claim<Self::F>> {
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::IndexedBit(round_index) {
                mle_index.bind_index(challenge);
            }
        }

        // --- One fewer iterated bit to sumcheck through ---
        self.num_vars -= 1;

        if self.num_vars == 0 {
            let mut send_claim = Claim::new_raw(
                self.mle_indices
                    .iter()
                    .map(|index| index.val().unwrap())
                    .collect_vec(),
                F::zero(),
            );
            send_claim.mle_ref = Some(MleEnum::Zero(self.clone()));
            Some(send_claim)
        } else {
            None
        }
    }

    fn fix_variable_at_index(
        &mut self,
        indexed_bit_index: usize,
        point: Self::F,
    ) -> Option<Claim<Self::F>> {
        self.fix_variable(indexed_bit_index, point)
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

    fn get_layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn push_mle_indices(&mut self, new_indices: &[MleIndex<Self::F>]) {
        self.mle_indices.append(&mut new_indices.to_vec());
    }

    fn get_enum(self) -> MleEnum<Self::F> {
        MleEnum::Zero(self)
    }
}
