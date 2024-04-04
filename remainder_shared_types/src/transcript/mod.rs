// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//!A type that is responsible for FS over the interative version of the protocol

use thiserror::Error;
pub mod poseidon_transcript;

///An error representing the things that can go wrong when working with a Transcript
#[derive(Error, Debug, Clone)]
pub enum TranscriptError {
    #[error("The challenges generated don't match challenges given!")]
    TranscriptMatchError,
}

///A type that is responsible for FS over the interative version of the protocol
pub trait Transcript<F>: Clone {
    ///Create an empty transcript
    fn new(label: &'static str) -> Self;

    ///Append a single field element to the transcript
    fn append_field_element(
        &mut self,
        label: &'static str,
        element: F,
    ) -> Result<(), TranscriptError>;

    ///Append a list of field elements to the transcript
    fn append_field_elements(
        &mut self,
        label: &'static str,
        elements: &[F],
    ) -> Result<(), TranscriptError>;

    ///Generate a random challenge and add it to the transcript
    fn get_challenge(&mut self, label: &'static str) -> Result<F, TranscriptError>;

    ///Generate a list of random challenges and add it to the transcript
    fn get_challenges(
        &mut self,
        label: &'static str,
        len: usize,
    ) -> Result<Vec<F>, TranscriptError>;
}
