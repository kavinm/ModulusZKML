use remainder::layer::LayerId;
use remainder::mle::bin_decomp_structs::bin_decomp_32_bit::BinDecomp32Bit;
use remainder::mle::MleIndex;
use remainder::{expression::ExpressionStandard, layer::LayerBuilder, mle::dense::DenseMle};
use remainder_shared_types::FieldExt;

/// NOTE: BIN DECOMP IS ASSUMED TO BE IN LITTLE-ENDIAN, WITH THE SIGN BIT AT THE END
///
/// This builder takes in a 32-bit signed decomp {b_0, ..., b_{30}, b_s} and outputs
/// the the positive recomposition.
pub struct BinaryRecompBuilder32Bit<F: FieldExt> {
    signed_bin_decomp: DenseMle<F, BinDecomp32Bit<F>>,
}
impl<F: FieldExt> LayerBuilder<F> for BinaryRecompBuilder32Bit<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        let bit_mle_refs = self.signed_bin_decomp.mle_bit_refs();

        // --- Let's just do a linear accumulator for now ---
        // TODO!(ryancao): Rewrite this expression but as a tree
        let b_s_initial_acc = ExpressionStandard::Constant(F::zero());

        bit_mle_refs
            .into_iter()
            // --- Get rid of sign bit ---
            .rev()
            .skip(1)
            .rev()
            .enumerate()
            .fold(b_s_initial_acc, |acc_expr, (bit_idx, bin_decomp_mle)| {
                // --- Coeff MLE ref (i.e. b_i) ---
                let b_i_mle_expression_ptr = Box::new(ExpressionStandard::Mle(bin_decomp_mle));

                // --- Compute (coeff) * 2^{bit_idx} ---
                let base = F::from(2_u64.pow(bit_idx as u32));
                let b_s_times_coeff_times_base =
                    ExpressionStandard::Scaled(b_i_mle_expression_ptr, base);

                acc_expr + b_s_times_coeff_times_base
            })
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let result_iter = self.signed_bin_decomp.into_iter().map(|signed_bin_decomp| {
            signed_bin_decomp
                .bits
                .into_iter()
                .rev()
                .skip(1)
                .rev()
                .enumerate()
                .fold(F::zero(), |acc, (bit_idx, cur_bit)| {
                    let base = F::from(2_u64.pow(bit_idx as u32));
                    acc + base * cur_bit
                })
        });

        DenseMle::new_from_iter(result_iter, id, prefix_bits)
    }
}
impl<F: FieldExt> BinaryRecompBuilder32Bit<F> {
    /// Constructor
    pub fn new(signed_bin_decomp: DenseMle<F, BinDecomp32Bit<F>>) -> Self {
        Self { signed_bin_decomp }
    }
}
