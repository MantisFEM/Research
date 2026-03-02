# Paper Metadata

Title: Construction of exact refinement for the two-dimensional hierarchical de Rham complex

Year: 2026

Authors: Diogo C. Cabanas, Kendrick M. Shepherd, Deepesh Toshniwal, and Rafael Vázquez

Links: [arXiv](https://arxiv.org/abs/2502.19542)

# Running the examples

All the numerical results, show in Section 7, in the paper were implemented as in the `.jl`
files included in this repository.

A helper script, `run_examples.sh`, is also included. This will instantiate the Julia
project, run all the examples, and create and populate an `exports` folder with the results.
Be sure to make the script executable by running `chmod +x run_examples.sh`.

For full reproducibility of the results we also include a `Manifest.toml`. If for some
reason this leads to issues, you can try to delete it and retry to instantiate the project
using just the `Project.toml`.
