// Copyright © 2024.  Modulus Labs, Inc.

// Restricted Use License

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ìSoftwareî), to use the Software internally for evaluation, non-production purposes only.  Any redistribution, reproduction, modification, sublicensing, publication, or other use of the Software is strictly prohibited.  In addition, usage of the Software is subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED ìAS ISî, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#[cfg(test)]
mod tests {
    use remainder::prover::GKRCircuit;
    use remainder_shared_types::{
        transcript::{poseidon_transcript::PoseidonTranscript, Transcript},
        Fr,
    };

    use crate::simple_arithmetic_circuit::{
        circuit::SimpleArithmeticCircuit,
        generate_random_inputs::{
            generate_random_inputs_for_simple_arithmetic_circuit, SimpleArithmeticCircuitInput,
        },
    };

    #[test]
    fn test_simple_arithmetic_circuit() {
        let data = generate_random_inputs_for_simple_arithmetic_circuit(4);

        let SimpleArithmeticCircuitInput::<Fr> {
            mle,
            two_times_mle_squared,
        } = (data).into();

        let mut transcript = PoseidonTranscript::new("Simple Arithmetic Circuit Transcript");
        let mut circuit = SimpleArithmeticCircuit::new(mle, two_times_mle_squared);
        // generate the proof from the circuit
        let proof = circuit.prove(&mut transcript);

        match proof {
            // use the GKR verifier to verify this proof. if the circuit is constructed correctly
            // and the inputs are provided correctly (which we handle :D) then the circuit should verify
            // and the test will pass!
            Ok(proof) => {
                let mut transcript =
                    PoseidonTranscript::new("Simple Arithmetic Circuit Transcript");
                let result = circuit.verify(&mut transcript, proof);
                if let Err(err) = result {
                    println!("{}", err);
                    panic!();
                }
            }
            Err(err) => {
                println!("{}", err);
                panic!();
            }
        }
    }
}
