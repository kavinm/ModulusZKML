// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use remainder_shared_types::FieldExt;
use remainder_shared_types::Poseidon;

/// FieldHashFnDigest gives you a trait which is basically `Digest` but compatible
/// with FieldExt-based hash functions.
pub trait FieldHashFnDigest<F: FieldExt> {
    /// Parameters to pass into `new_with_params` such that a digest instance
    /// with specific hash function parameters can be created.
    type HashFnParams;

    /// (FieldHashFnDigest) Create new hasher instance with default params
    fn new() -> Self;

    /// (FieldHashFnDigest) Create new hasher instance but with given params
    fn new_with_params(params: Self::HashFnParams) -> Self;

    /// (FieldHashFnDigest) Create new hasher instance with default params for Merkle hashing
    fn new_merkle_hasher(static_merkle_poseidon_hasher: &Poseidon<F, 3, 2>) -> Self;

    /// (FieldHashFnDigest) Create new hasher instance with default params for column hashing
    fn new_column_hasher(static_column_poseidon_hasher: &Poseidon<F, 3, 2>) -> Self;

    /// (FieldHashFnDigest) Digest data, updating the internal state.
    ///
    /// This method can be called repeatedly for use with streaming messages.
    fn update(&mut self, data: &[F]);

    /// (FieldHashFnDigest) Digest input data in a chained manner.
    fn chain(self, data: &[F]) -> Self
    where
        Self: Sized;

    /// (FieldHashFnDigest) Retrieve result and consume hasher instance.
    fn finalize(self) -> F;

    /// (FieldHashFnDigest) Retrieve result and reset hasher instance.
    ///
    /// This method sometimes can be more efficient compared to hasher
    /// re-creation.
    fn finalize_reset(&mut self) -> F;

    /// (FieldHashFnDigest) Reset hasher instance to its initial state.
    fn reset(&mut self);

    /// (FieldHashFnDigest) Get output size of the hasher
    fn output_size() -> usize;

    /// (FieldHashFnDigest) Convenience function to compute hash of the `data`. It will handle
    /// hasher creation, data feeding and finalization.
    fn digest(data: &[F]) -> F;
}
