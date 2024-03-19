use ark_std::log2;
use itertools::{repeat_n, Itertools};
use crate::{layer::{batched::combine_mles, combine_mle_refs::combine_mle_refs, LayerId}, mle::{dense::{get_padded_evaluations_for_list, DenseMle, DenseMleRef}, Mle, MleAble, MleIndex}};
use remainder_shared_types::FieldExt;
use serde::{Deserialize, Serialize};

/// --- 32-bit binary decomposition ---
#[derive(Copy, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinDecomp32Bit<F> {
    ///The 32 bits that make up this decomposition
    ///
    /// Should all be 1 or 0
    pub bits: [F; 32],
}

impl<F: FieldExt> From<Vec<bool>> for BinDecomp32Bit<F> {
    fn from(bits: Vec<bool>) -> Self {
        BinDecomp32Bit::<F> {
            bits: bits.iter().map(|x| F::from(*x as u64)).collect::<Vec<F>>().try_into().unwrap()
        }
    }
}

// --- Bin decomp ---
impl<F: FieldExt> MleAble<F> for BinDecomp32Bit<F> {
    type Repr = [Vec<F>; 32];

    fn get_padded_evaluations(items: &Self::Repr) -> Vec<F> {
        get_padded_evaluations_for_list(items)
    }

    type IntoIter<'a> = std::vec::IntoIter<BinDecomp32Bit<F>> where Self: 'a;

    fn from_iter(iter: impl IntoIterator<Item = Self>) -> Self::Repr {
        let iter = iter.into_iter();

        // --- TODO!(ryancao): This is genuinely horrible but we'll fix it later ---
        let mut ret: [Vec<F>; 32] = std::array::from_fn(|_| vec![]);
        iter.for_each(|tuple| {
            for (item, bit) in ret.iter_mut().zip(tuple.bits.iter()) {
                item.push(*bit);
            }
        });

        ret
    }

    fn to_iter(items: &Self::Repr) -> Self::IntoIter<'_> {
        let elems = (0..items[0].len()).map(
            |idx| {
                let bits = items.iter().map(
                    |item| {
                        item[idx]
                    }
                ).collect_vec();
                BinDecomp32Bit {
                    bits: bits.try_into().unwrap(),
                }
            }
        ).collect_vec();

        elems.into_iter()
    }

    fn num_vars(items: &Self::Repr) -> usize {
        log2(32 * items[0].len()) as usize
    }
}

// TODO!(ryancao): Make this stuff derivable
impl<F: FieldExt> DenseMle<F, BinDecomp32Bit<F>> {
    /// Returns a list of MLERefs, one for each bit
    /// TODO!(ryancao): Change this back to [DenseMleRef<F>; 32] and make it work!
    pub fn mle_bit_refs(&'_ self) -> Vec<DenseMleRef<F>> {
        let num_vars = self.num_iterated_vars();

        // --- There are sixteen components to this MLE ---
        let mut ret: Vec<DenseMleRef<F>> = vec![];

        for bit_idx in 0..32 {
            // --- Prefix bits need to be *literally* represented in little-endian ---
            let first_prefix = (bit_idx % 2) >= 1;
            let second_prefix = (bit_idx % 4) >= 2;
            let third_prefix = (bit_idx % 8) >= 4;
            let fourth_prefix = (bit_idx % 16) >= 8;
            let fifth_prefix = (bit_idx % 32) >= 16;

            let mle_indices = self
            .prefix_bits
            .clone()
            .into_iter()
            .flatten()
            .chain(
                std::iter::once(MleIndex::Fixed(first_prefix))
                    .chain(std::iter::once(MleIndex::Fixed(second_prefix)))
                    .chain(std::iter::once(MleIndex::Fixed(third_prefix)))
                    .chain(std::iter::once(MleIndex::Fixed(fourth_prefix)))
                    .chain(std::iter::once(MleIndex::Fixed(fifth_prefix)))
                    .chain(repeat_n(MleIndex::Iterated, num_vars - 5)),
            )
            .collect_vec();

            let bit_mle_ref = DenseMleRef {
                bookkeeping_table: self.mle[bit_idx].to_vec(),
                original_bookkeeping_table: self.mle[bit_idx].to_vec(),
                // --- [0, 0, 0, 0, 0, b_1, ..., b_n] ---
                mle_indices: mle_indices.clone(),
                original_mle_indices: mle_indices,
                num_vars: num_vars - 5,
                original_num_vars: num_vars - 5,
                layer_id: self.layer_id,
                indexed: false,
            };
            ret.push(bit_mle_ref);
        }

        ret
    }

    /// Returns the entire bin decomp MLE as a single MLE ref
    pub fn get_entire_mle_as_mle_ref(&'_ self) -> DenseMleRef<F> {
        // --- Just need to merge all of the bin decomps in an interleaved fashion ---
        // TODO!(ryancao): This is an awful hacky fix so that we can use `combine_mles`.
        // Note that we are manually inserting the extra iterated bits as prefix bits.
        // We should stop doing this once `combine_mles` works as it should!
        let self_mle_ref_vec = self.mle.clone().map(|mle_bookkeeping_table| {
            DenseMle::new_from_raw(
                mle_bookkeeping_table, 
                self.layer_id, 
                Some(
                    self.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, 5)
                    ).collect_vec()
                )
            ).mle_ref()
        }).to_vec();
        combine_mles(self_mle_ref_vec, 5)
    }

    /// Combines the bookkeeping tables of each of the MleRefs within a
    /// `DenseMle<F, BinDecomp32Bit<F>>` into a single interleaved bookkeeping
    /// table such that referring to the merged table using little-endian indexing
    /// bits, followed by the appropriate MleRef indexing bits, gets us the same
    /// result as only using the same MleRef indexing bits on each MleRef from
    /// the `DenseMle<F, BinDecomp32Bit<F>>`.
    /// 
    /// TODO!(ende): refactor
    pub fn combine_mle_batch(input_mle_batch: Vec<DenseMle<F, BinDecomp32Bit<F>>>) -> DenseMle<F, F> {
        
        let batched_bits = log2(input_mle_batch.len());

        let input_mle_batch_ref_combined = input_mle_batch
            
            .into_iter().map(
                |x| {
                    combine_mle_refs(
                        x.mle_bit_refs()
                    ).mle_ref()
                }
            ).collect_vec();

        let input_mle_batch_ref_combined_ref = combine_mles(input_mle_batch_ref_combined, batched_bits as usize);

        DenseMle::new_from_raw(input_mle_batch_ref_combined_ref.bookkeeping_table, LayerId::Input(0), None)
    }
}