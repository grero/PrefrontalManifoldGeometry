module FigureS5
using Makie
using CairoMakie
using ..PlotUtils
using ..Utils
using ..Figure2

"""
Show per-subject go-cue and movement onset subspaces, as well as these spaces for target-retarget trials in subject W.
"""
function plot(;max_latency=70.0)
    with_theme(PlotUtils.plot_theme) do
        fig = Figure(size=(500,500))
        lg1 = GridLayout(fig[1,1])
        lg2 = GridLayout(fig[2,1])
        lg3 = GridLayout(fig[3,1])
        
        #Monkey J
        fname_cue, fname_mov = Figure2.get_event_subspaces(;subject="J", rtime_min=120.0,area="FEF")
        Figure2.plot_event_onset_subspaces!(lg1, fname_cue, fname_mov;max_latency=max_latency,xticklabelsvisible=false, xlabelvisible=false,titlevisible=true)
        Label(fig[1,1,TopLeft()],"A")

        #Monkey W
        fname_cue, fname_mov = Figure2.get_event_subspaces(;subject="W", rtime_min=120.0,area="FEF")
        Figure2.plot_event_onset_subspaces!(lg2, fname_cue, fname_mov;max_latency=max_latency,xticklabelsvisible=false, xlabelvisible=false,titlevisible=false)
        Label(fig[2,1,TopLeft()],"B")

        #Monkey w target-retarget trials
        fname_cue, fname_mov = Figure2.get_event_subspaces(;subject="ALL", rtime_min=120.0,area="FEF",suffix="_tt")
        Figure2.plot_event_onset_subspaces!(lg3, fname_cue, fname_mov;max_latency=max_latency, titlevisible=false)
        Label(fig[3,1,TopLeft()],"C")
        fig
    end
end
end #module