# Prefrontal Manifold geometry
Codes to reproduce the main figures in the paper "Prefrontal Manifold Geometry Contributes to Reaction Time Variability".

## Installation

1. Install the Julia programming language. The results were produced using version 1.9.3, which can be downloaded from https://julialang.org/downloads/oldreleases/. The latest release, available from https://julialang.org/downloads/ should also work. For instructions on how to install Julia on the various platforms, see https://julialang.org/downloads/platform/
2. Add the package repository containing this code, as well as supporting packages by running this from the julia prompt

```julia
(@v1.9) pkg> registry add https://github.com/grero/NeuralCodingRegistry.jl.git
```
To get to the add `pkg>` prompt, press `]` in the REPL

2. Clone this repository using e.g. bash

```bash
git clone https://github.com/grero/PrefrontalManifoldGeometry.git PrefrontalManifoldGeometry
```
3. Start Julia from the newly cloned folder

```bash
cd PrefrontalManifoldGeometry
julia --project=.
```

or, from within the REPL itself, navigate to the folder, enter package mode, and type `activate .`, followed by `instantiate`. The latter command will install all  package dependencies.

```julia
shell> cd PrefrontalManifoldGeometry
(@v1.9) pkg> activate .
(@v1.9) pkg> instantiate 
```
to enter shell mode, press `;` in the REPL.

## Usage

The codes for producing each of the main figures are organized in their own module. This means that to reproduce for example figure 1, you can execute the following from the Julia REPL:

```julia
using PrefrontalManifoldGeometry
const PMG = PrefrontalManifoldGeometry
using Makie, CairoMakie
fig1 = PMG.Figure1.plot()
```