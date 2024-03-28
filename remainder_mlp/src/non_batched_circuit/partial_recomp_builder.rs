use remainder::{expression::ExpressionStandard, layer::LayerBuilder, mle::{bin_decomp_structs::bin_decomp_64_bit::BinDecomp64Bit, dense::DenseMle, structs::BinDecomp16Bit, zero::ZeroMleRef, Mle}};
use remainder_shared_types::FieldExt;
use remainder::layer::LayerId;
use remainder::mle::MleIndex;


/// NOTE: BIN DECOMP IS ASSUMED TO BE IN LITTLE-ENDIAN, WITH THE SIGN BIT AT THE END
/// 
/// This builder takes in a 64-bit signed decomp {b_0, ..., b_{62}, b_s} and outputs
/// the "un-scaled" version of the positive recomposition. In other words:
/// 
/// * Define the "positive recomposition" as p = \sum_{i = 0}^{62} b_i 2^i
/// * The value we output is thus y = \lfloor p / 2^{62 - `recomp_bitwidth`} \rfloor
/// * Equivalently, this is y = \sum_{i = 62 - `recomp_bitwidth` + 1}^{62} b_i 2^i
/// 
/// TODO!(ryancao): We should be able to do this easily for 32-bit stuff as well!
pub struct PartialPositiveBinaryRecompBuilder<F: FieldExt> {
    signed_bin_decomp: DenseMle<F, BinDecomp64Bit<F>>,
    recomp_bitwidth: usize,
}
impl<F: FieldExt> LayerBuilder<F> for PartialPositiveBinaryRecompBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        let bit_mle_refs = self.signed_bin_decomp.mle_bit_refs();

        // --- Let's just do a linear accumulator for now ---
        // TODO!(ryancao): Rewrite this expression but as a tree
        let b_s_initial_acc = ExpressionStandard::Constant(F::zero());

        // --- We want to iterate through the last (most significant) `recomp_bitwidth` - 1 bits (save one bit for b_s) ---

        bit_mle_refs.into_iter().take(self.recomp_bitwidth - 1).enumerate().fold(
            b_s_initial_acc,
            |acc_expr, (bit_idx, bin_decomp_mle)| {

                // --- Coeff MLE ref (i.e. b_i) ---
                let b_i_mle_expression_ptr = Box::new(ExpressionStandard::Mle(bin_decomp_mle));

                // --- Compute (coeff) * 2^{63 - bit_idx} ---
                let base = F::from(2_u64.pow(bit_idx as u32));
                let b_s_times_coeff_times_base =
                    ExpressionStandard::Scaled(b_i_mle_expression_ptr, base);

                acc_expr + b_s_times_coeff_times_base
            },
        )
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {

        let result_iter = self.signed_bin_decomp.into_iter().map(
            |signed_bin_decomp| {
                signed_bin_decomp.bits.into_iter().take(self.recomp_bitwidth - 1).enumerate().fold(F::zero(), |acc, (bit_idx, cur_bit)| {
                    let base = F::from(2_u64.pow(bit_idx as u32));
                    acc + base * cur_bit
                })
            }
        );
        
        DenseMle::new_from_iter(result_iter, id, prefix_bits)
    }
}
impl<F: FieldExt> PartialPositiveBinaryRecompBuilder<F> {
    /// Constructor
    pub fn new(
        diff_signed_bin_decomp: DenseMle<F, BinDecomp64Bit<F>>, 
        recomp_bitwidth: usize
    ) -> Self {
        Self {
            signed_bin_decomp: diff_signed_bin_decomp,
            recomp_bitwidth
        }
    }
}