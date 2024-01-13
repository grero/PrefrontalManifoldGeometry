# Prefrontal Manifold geometry
Codes to reproduce the main figures in the paper "Prefrontal Manifold Geometry Explains Reaction Time Variability".

## Installation

1. Install the Julia programming language. The results were produced using version 1.9.3, which can be downloaded from https://julialang.org/downloads/oldreleases/. The latest release, available from https://julialang.org/downloads/ should also work. For instructions on how to install Julia on the various platforms, see https://julialang.org/downloads/platform/
2. Add the package repository containing this code, as well as supporting packages by running this from the julia prompt

```julia
(@v1.9) pkg> registry add https://github.com/grero/NeuralCodingRegistry.jl.git
```
To get to the add `pkg>` prompt, press `]` in the REPL

2. It is recommended that you create a new environment in which to install the various packages (similar to a `conda` environment in Python). To do, either start Julia from the directory containing the environment, e.g. `$HOME/Documents/PrefrontalManifoldGeometryPaperTest`,


```bash
julia --project=.
```

or, from within the REPL itself, navigate to the directory, enter package mode, and type `activate .`

```julia
shell> cd $HOME/Documents/PrefrontalManifoldGeometryPaperTest
(@v1.9) pkg> activate .
```
to enter shell mode, press `;` in the REPL.

Once you have activated the environment, add this package using

```julia
(PrefrontalManifoldGeometryPaperTest) pkg> add PrefronalManifoldGeometry
```

## Usage

The codes for producing each of the main figures are organized in their own module. This means that to reproduce for example figure 1, you can execute the following from the Julia REPL:

```julia
using PrefrontalManifoldGeometry
const PMG = PrefrontalManifoldGeometry
using PMG: Makie, CairoMakie
using Mkie, CairoMakie
fig1 = PMG.Figure1.plot()
```