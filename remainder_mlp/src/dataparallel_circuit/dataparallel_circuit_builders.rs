use remainder::{expression::ExpressionStandard, layer::LayerBuilder, mle::{bin_decomp_structs::bin_decomp_64_bit::BinDecomp64Bit, dense::DenseMle, structs::BinDecomp16Bit, zero::ZeroMleRef, Mle}};
use remainder_shared_types::FieldExt;
use remainder::layer::LayerId;
use remainder::mle::MleIndex;


/// For asserting that the binary decomposition of the bits which purportedly
/// make up the difference between the node's threshold value and the actual
/// attribute's value on the path recomposes (in a bitwise fashion) to create
/// the actual difference.
/// 
/// This builder computes the positive binary recomposition
/// \sum_{b_1, ... b_{15}} 2^{15 - i} * b_i
pub struct PositiveBinaryRecompBuilder<F: FieldExt> {
    diff_signed_bin_decomp: DenseMle<F, BinDecomp16Bit<F>>,
}
impl<F: FieldExt> LayerBuilder<F> for PositiveBinaryRecompBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        let bit_mle_refs = self.diff_signed_bin_decomp.mle_bit_refs();

        // --- Let's just do a linear accumulator for now ---
        // TODO!(ryancao): Rewrite this expression but as a tree
        let b_s_initial_acc = ExpressionStandard::Constant(F::zero());

        bit_mle_refs.into_iter().rev().enumerate().skip(1).fold(
            b_s_initial_acc,
            |acc_expr, (bit_idx, bin_decomp_mle)| {

                // --- Coeff MLE ref (i.e. b_i) ---
                let b_i_mle_expression_ptr = Box::new(ExpressionStandard::Mle(bin_decomp_mle));

                // --- Compute (coeff) * 2^{15 - bit_idx} ---
                let base = F::from(2_u64.pow((16 - (bit_idx + 1)) as u32));
                let b_s_times_coeff_times_base =
                    ExpressionStandard::Scaled(b_i_mle_expression_ptr, base);

                acc_expr + b_s_times_coeff_times_base
            },
        )
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {

        let result_iter = self.diff_signed_bin_decomp.into_iter().map(
            |signed_bin_decomp| {
                signed_bin_decomp.bits.into_iter().rev().enumerate().skip(1).fold(F::zero(), |acc, (bit_idx, cur_bit)| {
                    let base = F::from(2_u64.pow((16 - (bit_idx + 1)) as u32));
                    acc + base * cur_bit
                })
            }
        );
        
        DenseMle::new_from_iter(result_iter, id, prefix_bits)
    }
}
impl<F: FieldExt> PositiveBinaryRecompBuilder<F> {
    /// Constructor
    pub fn new(diff_signed_bin_decomp: DenseMle<F, BinDecomp16Bit<F>>) -> Self {
        Self {
            diff_signed_bin_decomp
        }
    }
}


/// For asserting that the binary decomposition of the bits which purportedly
/// make up the difference between the node's threshold value and the actual
/// attribute's value on the path recomposes (in a bitwise fashion) to create
/// the actual difference.
/// 
/// This builder computes the value `pos_recomp` - `diff` + 2 * `sign_bit` * `diff`.
/// Note that this is equivalent to
/// (1 - b_s)(`pos_recomp` - `diff`) + `b_s`(`pos_recomp` + `diff`)
pub struct BinaryRecompCheckerBuilder<F: FieldExt> {
    mle: DenseMle<F, F>,
    signed_bit_decomp_mle: DenseMle<F, BinDecomp64Bit<F>>,
    positive_recomp_mle: DenseMle<F, F>,
}
impl<F: FieldExt> LayerBuilder<F> for BinaryRecompCheckerBuilder<F> {
    type Successor = ZeroMleRef<F>;

    fn build_expression(&self) -> ExpressionStandard<F> {

        // --- Grab MLE refs ---
        let positive_recomp_mle_ref = self.positive_recomp_mle.mle_ref();
        let signed_bit_mle_ref = self.signed_bit_decomp_mle.mle_bit_refs()[self.signed_bit_decomp_mle.mle_bit_refs().len() - 1].clone();
        let diff_mle_ref = self.mle.mle_ref();

        // --- LHS of addition ---
        let pos_recomp_minus_diff = ExpressionStandard::Mle(positive_recomp_mle_ref) - ExpressionStandard::Mle(diff_mle_ref.clone());

        // --- RHS of addition ---
        let sign_bit_times_diff_ptr = Box::new(ExpressionStandard::Product(vec![signed_bit_mle_ref, diff_mle_ref]));
        let two_times_sign_bit_times_diff = ExpressionStandard::Scaled(sign_bit_times_diff_ptr, F::from(2));

        pos_recomp_minus_diff + two_times_sign_bit_times_diff
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {

        ZeroMleRef::new(self.positive_recomp_mle.num_iterated_vars(), prefix_bits, id)
    }
}
impl<F: FieldExt> BinaryRecompCheckerBuilder<F> {
    /// Constructor
    pub fn new(
        mle: DenseMle<F, F>,
        signed_bit_decomp_mle: DenseMle<F, BinDecomp64Bit<F>>,
        positive_recomp_mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            mle,
            signed_bit_decomp_mle,
            positive_recomp_mle
        }
    }
}