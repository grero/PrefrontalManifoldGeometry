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

The codes for producing each of the main figures are organized in their own module. This means that to reproduce all the figures, you can execute the following from the Julia REPL:

```julia
using PrefrontalManifoldGeometry
const PMG = PrefrontalManifoldGeometry
using CairoMakie
fig1 = PMG.Figure1.plot()
fig2 = PMG.Figure2.plot()
fig3 = PMG.Figure3.plot()
fig4 = PMG.Figure4.plot()
fig5 = PMG.Figure5.plot()
```

As some of the analyses take a long time to run, each of the plot functions will load pre-computed data by default. To re-run any analysis from scrtach (i.e. starting from single cell spike counts), use the argument `redo=true` to any function.

### Detailed usage

A detailed explnataion of the code used to find the onset and offset of the transition period from post-go-cue to pre-movement (Figure 2a) can be found [here](https://github.com/grero/EventOnsetDecoding.jl/blob/main/notebooks/basic_tutorial.ipynb).

This code reproduces Figure 2a from population spike count data:

```julia
using JLD2
using PrefrontalManifoldGeometry
const PMG = PrefrontalManifoldGeometry
using EventOnsetDecoder
using EventOnsetDecoder: MultivariateStats

# load spike counts for both go-cue and movement aligned trials
ppsth_cue,labels_cue, trialidx_cue, rtimes_cue = JLD2.load("data/ppsth_fef_cue.jld2","ppsth", "labels","trialidx","rtimes")
ppsth_mov,labels_mov, trialidx_mov, rtimes_mov = JLD2.load("data/ppsth_fef_mov.jld2","ppsth", "labels","trialidx","rtimes")

sessions = [PMG.sessions_j;PMG_sessions_w]
locations = [1:8;]
cellidx = 1:size(ppsth.counts,3)
args = [sessions, locations, cellidx]
latencies = range(100.0, step=-10.0, stop=0.0)
windows = [range(5.0, step=10.0, stop=50.0);]
kvs = [:runs => 100,:difference_decoder=>true, :windows=>windows,:latencies=>latencies, :combine_locations=true, :use_area=>"ALL",
       rtime_min=120.0, :mixin_postcue=>true]

dargs_cue = EventOnsetDecoding.DecoderArgs(args...;kvs..., reverse_bins=true, baseline_end=-250.0)
dargs_mov = EventOnsetDecoding.DecoderArgs(args...;kvs..., reverse_bins=false, baseline_end=-300.0)
rseeds = rand(UInt32, dargs_mov.nruns)
# train cue and movement onset decoders
perf_cue,rr_cue,f1score_cue,fname_cue = EventOnsetDecoding.run_rtime_decoder(ppsth_cue,trialidx_cue,labels_cue,rtimes_cue,dargs_cue,
                                                         ;decoder=MultivariateStats.MulticlassLDA, rseeds=rseeds, redo=true)
perf_mov,rr_mov,f1score_mov,fname_mov = EventOnsetDecoding.run_rtime_decoder(ppsth_mov,trialidx_mov,labels_mov,rtimes_mov,dargs_mov,
                                                               ;decoder=MultivariateStats.MulticlassLDA, rseeds=rseeds, redo=true)

# perf is the raw performance of the decoder, while f1score is performance adjusted for both false positives and false negatives.
# in Figure 2a, we plot f1-score instead of performance,
# plot
fig = plot_event_onset_subspaces(fname_cue, fname_mov)
```
