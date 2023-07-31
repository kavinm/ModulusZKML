use crate::FieldExt;
use crate::layer::{LayerBuilder, LayerId};
use crate::expression::{ExpressionStandard }; 
use crate::mle::{MleIndex, Mle};
use crate::mle::dense::{DenseMle, Tuple2};


struct ProductTreeBuilder<F:FieldExt> {
    mle: DenseMle<F, Tuple2<F>>,
} 

impl<F: FieldExt> LayerBuilder<F> for ProductTreeBuilder<F> { 

    type Successor = DenseMle<F, F>;
    //a function that multiplies the parts of the tuple pair-wise
    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::products(vec![self.mle.first(), self.mle.second()])
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        // Create flatmle from tuple mle
        let mut flat_mle: DenseMle<F, F> = self.mle.into_iter().map( |(first, second)| first * second).collect();
        flat_mle.add_prefix_bits(&prefix_bits.unwrap());
        flat_mle
    }

}





