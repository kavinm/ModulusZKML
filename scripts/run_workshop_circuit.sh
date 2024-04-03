cargo build --release
cargo run --release --bin run_workshop_circuit -- \
    --input-dim 784 \
    --output-dim 200 \
    --hidden-dim 10 \
    --verify-proof \