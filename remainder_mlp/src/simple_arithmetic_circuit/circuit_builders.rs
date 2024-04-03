use itertools::Itertools;
use remainder::{
    expression::ExpressionStandard,
    layer::{LayerBuilder, LayerId},
    mle::{
        dense::{DenseMle, DenseMleRef},
        zero::ZeroMleRef,
        MleIndex,
    },
};
use remainder_shared_types::FieldExt;

// ------------------------ MLE Squaring Builder ------------------------

/// This circuit has a single MLE as input -- that which is to be squared
/// element-wise over b_1, ..., b_n \in \{0, 1\}^n.
pub struct MleSquaringBuilder<F: FieldExt> {
    mle_to_be_squared_ref: DenseMleRef<F>,
}

/// The LayerBuilder trait requires you to implement two functions:
/// the build expression function and the next layer function
impl<F: FieldExt> LayerBuilder<F> for MleSquaringBuilder<F> {
    /// The `Self::Successor` type defines the output type of the `next_layer`
    /// function, i.e. the type of MleRef which this layer computes from its
    /// input MLEs.
    type Successor = DenseMleRef<F>;

    /// The `build_expression` function returns an expression representing the
    /// polynomial relationship between the input MLEs (i.e. those present
    /// within the `struct MleSquaringBuilder`) and the output of the
    /// `MleSquaringBuilder` circuit layer.
    ///
    /// In this case, let `mle_to_be_squared`, represented by
    /// \tilde{V}_{i + 1}(b_1, ..., b_n), be the input MLE. The output is then
    /// the MLE derived by extending the vector of the element-wise squares of
    /// V_{i + 1}(b_1, ..., b_n) evaluated over b_1, ..., b_n \in \{0, 1}^n,
    /// or equivalently,
    ///
    /// \tilde{V}_i(c_1, ..., c_n) = \sum_{b_1, ..., b_n} \beta((c_1, ..., c_n), (b_1, ..., b_n)) * V_{i + 1}(b_1, ..., b_n)^2
    fn build_expression(&self) -> ExpressionStandard<F> {
        // Note that in Rust, not putting the `return` keyword AND not putting
        // a semicolon after the final statement within a block causes that
        // statement's return value to be the block's default return value.
        ExpressionStandard::Product(vec![
            self.mle_to_be_squared_ref.clone(),
            self.mle_to_be_squared_ref.clone(),
        ])
    }

    /// The `next_layer` function performs the actual "computation" of the
    /// circuit, i.e. in this case it takes the data present within
    /// `mle_to_be_squared` and returns a `DenseMleRef` representing the
    /// element-wise square of all the values within `mle_to_be_squared`.
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        // --- Grabs the evaluations of `self.mle_to_be_squared_ref` over \{0, 1\}^n ---
        let bookkeeping_table = self.mle_to_be_squared_ref.bookkeeping_table.clone();

        // --- Goes through each evaluation and squares it, returning another vector with such entries ---
        let bookkeeping_table_element_wise_squared =
            bookkeeping_table.into_iter().map(|x| x * x).collect_vec();

        // --- Creates an MLE from the computed bookkeeping table and returns its associated MleRef ---
        // --- Prefix bits can always safely be set to `None`, and the `layer_id` is irrelevant as well ---
        DenseMle::new_from_raw(bookkeeping_table_element_wise_squared, id, prefix_bits).mle_ref()
    }
}

impl<F: FieldExt> MleSquaringBuilder<F> {
    /// Constructor helper function
    pub fn new(mle_to_be_squared_ref: DenseMleRef<F>) -> Self {
        Self {
            mle_to_be_squared_ref,
        }
    }
}

// ------------------------ Double and subtract builder ------------------------

/// This circuit has two MLEs as inputs, where one is to be multiplied by two
/// and subtracted from the other
pub struct DoubleAndSubtractBuilder<F: FieldExt> {
    to_be_doubled_mle_ref: DenseMleRef<F>,
    to_be_subtracted_mle_ref: DenseMleRef<F>,
}

impl<F: FieldExt> LayerBuilder<F> for DoubleAndSubtractBuilder<F> {
    /// In this case, the `Self::Successor` is a `ZeroMleRef`, since the output
    /// of the double-and-subtract layer should be the MLE whose evaluations are
    /// zero everywhere.
    type Successor = ZeroMleRef<F>;

    /// Let X(b_1, ..., b_n) represent the MLE to be doubled and W(b_1, ..., b_n)
    /// represent the MLE to be subtracted. Then our expression is
    ///
    /// V_i(b_1, ..., b_n) = 2 * X(b_1, ..., b_n) - W(b_1, ..., b_n)
    ///
    /// (note that X is just V^2 from `circuits.rs`)!
    fn build_expression(&self) -> ExpressionStandard<F> {
        // --- Creates a Box pointer around an `ExpressionStandard` wrapper
        // (called `ExpressionStandard::Mle`) for `MleRef` ---
        let to_be_doubled_mle_ref_expr_ptr =
            Box::new(ExpressionStandard::Mle(self.to_be_doubled_mle_ref.clone()));

        // --- Represents the actual expression ---
        ExpressionStandard::Scaled(to_be_doubled_mle_ref_expr_ptr, F::from(2))
            - ExpressionStandard::Mle(self.to_be_subtracted_mle_ref.clone())
    }

    /// In this case, we should have that the output values from this layer are
    /// all zero, and thus we return a `ZeroMleRef`.
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        // --- The number of variables present in the final MLE, i.e. the "n" in b_1, ..., b_n ---
        // --- This should be the same as the number of variables present in either of the ---
        // --- input MLEs ---
        let zero_mle_ref_num_vars = self.to_be_doubled_mle_ref.num_vars;

        // --- This is the return statement of the function ---
        ZeroMleRef::new(zero_mle_ref_num_vars, prefix_bits, id)
    }
}

impl<F: FieldExt> DoubleAndSubtractBuilder<F> {
    /// Constructor helper function
    pub fn new(
        to_be_doubled_mle_ref: DenseMleRef<F>,
        to_be_subtracted_mle_ref: DenseMleRef<F>,
    ) -> Self {
        Self {
            to_be_doubled_mle_ref,
            to_be_subtracted_mle_ref,
        }
    }
}
