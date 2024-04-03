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
