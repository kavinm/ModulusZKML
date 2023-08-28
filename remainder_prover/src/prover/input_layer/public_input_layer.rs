//! An input layer that is sent to the verifier in the clear

use std::marker::PhantomData;

use remainder_shared_types::{FieldExt, transcript::{Transcript, TranscriptError}};

use crate::{mle::{dense::DenseMle, MleRef}, layer::{LayerId, Claim}};

use super::{InputLayer, InputLayerError, MleInputLayer, enum_input_layer::InputLayerEnum};

///An Input Layer that is send to the verifier in the clear
pub struct PublicInputLayer<F: FieldExt, Tr> {
    mle: DenseMle<F, F>,
    layer_id: LayerId,
    _marker: PhantomData<Tr>
}

impl<F: FieldExt, Tr: Transcript<F>> InputLayer<F> for PublicInputLayer<F, Tr> {
    type Transcript = Tr;

    type Commitment = Vec<F>;

    type OpeningProof = ();

    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        Ok(self.mle.mle.clone())
    }

    fn append_commitment_to_transcript(commitment: &Self::Commitment, transcript: &mut Self::Transcript) -> Result<(), TranscriptError> {
        transcript.append_field_elements("Public Input Commitment", &commitment)
    }

    fn open(&self, _: &mut Self::Transcript, _: crate::layer::Claim<F>) -> Result<Self::OpeningProof, super::InputLayerError> {
        Ok(())
    }

    fn verify(commitment: &Self::Commitment, opening_proof: &Self::OpeningProof, claim: Claim<F>, transcript: &mut Self::Transcript) -> Result<(), super::InputLayerError> {
        let mut mle_ref = DenseMle::<F, F>::new_from_raw(commitment.clone(), LayerId::Input(0), None).mle_ref();
        mle_ref.index_mle_indices(0);

        let mut curr_bit = 0;
        let mut eval = None;
        for &chal in claim.0.iter() {
            if chal != F::one() && chal != F::zero() {
                eval = mle_ref.fix_variable(curr_bit, chal);
                curr_bit += 1;
            }
        }

        let eval = eval.ok_or(InputLayerError::PublicInputVerificationFailed)?;

        if eval == claim {
            Ok(())
        } else {
            Err(InputLayerError::PublicInputVerificationFailed)
        }
    }

    fn layer_id(&self) -> &LayerId {
        &self.layer_id
    }

    fn get_padded_mle(&self) -> DenseMle<F, F> {
        self.mle.clone()
    }

    fn to_enum(self) -> InputLayerEnum<F, Self::Transcript> {
        InputLayerEnum::PublicInputLayer(self)
    }
}

impl<F: FieldExt, Tr: Transcript<F>> MleInputLayer<F> for PublicInputLayer<F, Tr> {
    fn new(mle: DenseMle<F, F>, layer_id: LayerId) -> Self {
        Self {
            mle,
            layer_id,
            _marker: PhantomData
        }
    }
}