// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use itertools::Itertools;
use rand::Rng;
use remainder::{
    layer::LayerId,
    mle::{bin_decomp_structs::bin_decomp_8_bit::BinDecomp8Bit, dense::DenseMle},
};
use remainder_shared_types::FieldExt;

pub struct CircuitInput<F: FieldExt> {
    pub sign_bit_mle: DenseMle<F, F>,
    pub abs_decomp_mle: DenseMle<F, BinDecomp8Bit<F>>,
    pub signed_value_mle: DenseMle<F, F>,
}

pub struct SimpleArithmeticCircuitInput<F: FieldExt> {
    pub mle: DenseMle<F, F>,
    pub two_times_mle_squared: DenseMle<F, F>,
}
pub fn generate_random_inputs_for_simple_arithmetic_circuit<F: FieldExt>(
    num_vars: usize,
) -> SimpleArithmeticCircuitInput<F> {
    let mut rng = rand::thread_rng();
    let mle_bookkeeping_table = (0..2_usize.pow(num_vars as u32))
        .into_iter()
        .map(|_| F::from(rng.gen_range(0..1000)))
        .collect_vec();
    let two_times_mle_squared_bookkeeping_table = mle_bookkeeping_table
        .clone()
        .into_iter()
        .map(|x| F::from(2) * (x * x))
        .collect_vec();
    let mle: DenseMle<F, F> =
        DenseMle::new_from_raw(mle_bookkeeping_table, LayerId::Input(0), None);
    let two_times_mle_squared: DenseMle<F, F> = DenseMle::new_from_raw(
        two_times_mle_squared_bookkeeping_table,
        LayerId::Input(0),
        None,
    );
    SimpleArithmeticCircuitInput {
        mle,
        two_times_mle_squared,
    }
}
