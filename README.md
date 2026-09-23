# Paper Metadata

- Title: Weakly-$C^1$ solutions to the biharmonic problem on multi-patch domains
- Year: 2026
- Authors: Joey Dekker, Artur Palha, and Deepesh Toshniwal
- Links: [article](https://doi.org/10.1016/j.cam.2026.118179)

# Running the examples

The research conducted for this paper was started before `Mantis` became publicly available.
Hence, the code provided in this branch includes the required (older) `Mantis` source code.
All the numerical results, shown in section 5 of the paper, are available as `.jl` files in this branch.

Note that cloning this repository will include all branches, so, if you are only interested in running the examples in this branch you can use 
```bash
  git clone --single-branch --branch=paper/2026/Weakly-C1 https://github.com/MantisFEM/Research.git.
```

To run an example (replace '#-name' by the appropriate part of the filename):

```bash
  julia --project TestCase#-Name.jl
```

This will produce a csv with the computed data.

To plot the data (replace '#' by the number of the test case):

```bash
  julia --project -i PlotTestCase#.jl
```

This will open the Julia REPL after setting things up. Once this is open, you can use

```julia
  julia> display(figL2) # or figH1, figH2, figjump. TestCase5 only has a figH1.
```

to view the created figures.


For full reproducibility of the results we also include a `Manifest.toml`. 
If, for some reason, this leads to issues, you can try to delete it and retry to instantiate the project using just the Project.toml.
The code provided in this branch was last run using Julia 1.13.0.
