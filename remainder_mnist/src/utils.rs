use ark_std::test_rng;
use itertools::Itertools;
use rand::Rng;
use remainder::{layer::LayerId, mle::{dense::DenseMle, structs::BinDecomp16Bit}};
use remainder_shared_types::FieldExt;



pub fn recompute_16_bit_decomp<F: FieldExt>(
    decomp_bits: &[F; 16],
) -> F {
    // skip 1 because the last bit is the signed bit
    decomp_bits.iter().rev().enumerate().skip(1).fold(
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