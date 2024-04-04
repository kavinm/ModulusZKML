// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use serde::{Deserialize, Serialize};

use remainder_shared_types::FieldExt;

use super::{dense::DenseMleRef, zero::ZeroMleRef, Mle, MleIndex, MleRef};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MleEnum<F> {
    Dense(DenseMleRef<F>),
    Zero(ZeroMleRef<F>),
}

impl<F: FieldExt> MleRef for MleEnum<F> {
    type F = F;

    fn bookkeeping_table(&self) -> &[Self::F] {
        match self {
            MleEnum::Dense(item) => item.bookkeeping_table(),
            MleEnum::Zero(item) => item.bookkeeping_table(),
        }
    }

    fn original_bookkeeping_table(&self) -> &Vec<Self::F> {
        match self {
            MleEnum::Dense(item) => item.original_bookkeeping_table(),
            MleEnum::Zero(item) => item.original_bookkeeping_table(),
        }
    }

    fn mle_indices(&self) -> &[super::MleIndex<Self::F>] {
        match self {
            MleEnum::Dense(item) => item.mle_indices(),
            MleEnum::Zero(item) => item.mle_indices(),
        }
    }

    fn original_mle_indices(&self) -> &Vec<super::MleIndex<Self::F>> {
        match self {
            MleEnum::Dense(item) => item.original_mle_indices(),
            MleEnum::Zero(item) => item.original_mle_indices(),
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            MleEnum::Dense(item) => item.num_vars(),
            MleEnum::Zero(item) => item.num_vars(),
        }
    }

    fn original_num_vars(&self) -> usize {
        match self {
            MleEnum::Dense(item) => item.original_num_vars(),
            MleEnum::Zero(item) => item.original_num_vars(),
        }
    }

    fn fix_variable(
        &mut self,
        round_index: usize,
        challenge: Self::F,
    ) -> Option<crate::layer::claims::Claim<Self::F>> {
        match self {
            MleEnum::Dense(item) => item.fix_variable(round_index, challenge),
            MleEnum::Zero(item) => item.fix_variable(round_index, challenge),
        }
    }

    fn fix_variable_at_index(
        &mut self,
        indexed_bit_index: usize,
        point: Self::F,
    ) -> Option<crate::layer::claims::Claim<Self::F>> {
        match self {
            MleEnum::Dense(item) => item.fix_variable_at_index(indexed_bit_index, point),
            MleEnum::Zero(item) => item.fix_variable_at_index(indexed_bit_index, point),
        }
    }

    fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        match self {
            MleEnum::Dense(item) => item.index_mle_indices(curr_index),
            MleEnum::Zero(item) => item.index_mle_indices(curr_index),
        }
    }

    fn get_layer_id(&self) -> crate::layer::LayerId {
        match self {
            MleEnum::Dense(item) => item.get_layer_id(),
            MleEnum::Zero(item) => item.get_layer_id(),
        }
    }

    fn indexed(&self) -> bool {
        match self {
            MleEnum::Dense(item) => item.indexed(),
            MleEnum::Zero(item) => item.indexed(),
        }
    }

    fn get_enum(self) -> MleEnum<Self::F> {
        self
    }

    fn push_mle_indices(&mut self, new_indices: &[MleIndex<Self::F>]) {
        match self {
            MleEnum::Dense(item) => item.push_mle_indices(new_indices),
            MleEnum::Zero(item) => item.push_mle_indices(new_indices),
        }
    }
}
