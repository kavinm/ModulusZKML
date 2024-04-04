# Remainder Workshop
- [Remainder Workshop](#remainder-workshop)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Example](#example)
  - [Workshop](#workshop)

## Introduction
Welcome to a demo version of Remainder, Modulus Labs' GKR prover for ZKML
models! Note that this is just a demo version and does not contain fancy
bells and whistles, and should be used for educational purposes only!

## Installation
* First, follow the instructions from the [official Rust
documentation](https://doc.rust-lang.org/book/ch01-01-installation.html) to
install Rust. 
* Then from this directory (`remainder/`), run `cargo check`. There will be a
  mountain of warnings, but the repository should compile just fine!
* Finally, run `cargo test --release test_simple_arithmetic_circuit`. You should
  see output which confirms that a GKR proof was generated for the simple
  arithmetic circuit found within
  `remainder_mlp/src/simple_arithmetic_circuit/circuit.rs`.


## Example
To run an example of the primary circuit (multi-layer perceptron, i.e. a
feedforward neural network with ReLU as the only non-linearity), use the
following command (this emulates a MNIST model with 96% accuracy!):
```
cargo build --release
cargo run --release --bin run_soln_workshop_circuit -- \
    --input-dim 784 \
    --output-dim 200 \
    --hidden-dim 10 \
    --verify-proof \
```

## Workshop
For the hands-on code-writing, you will be working in
`remainder_mlp/src/workshop_exercise_circuit`! You will need to modify the
following files:
* The primary neural network circuit can be found in `nn_full_circuit.rs`
* The `bits_are_binary_builder.rs` is the first builder you will have to create
* The `relu_builder.rs` is the second builder you will have to create