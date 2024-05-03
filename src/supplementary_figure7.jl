module FigureS7
using CairoMakie
using HypothesisTests
using HDF5
using StatsBase

include("utils.jl")
include("plot_utils.jl")
include("figure2.jl")

"""
Comparing F1 score for at the onset of the go-cue subspace and the movement subspace between
FEF and DLPFC
"""
function plot_fef_dlpfc_comparison(;latency::Dict=Dict("cue"=>40.0,"mov"=>0.0),window::Dict=Dict("cue"=>15.0, "mov"=>35.0),plottype=:scatter)
    fname_cue_dlpfc, fname_mov_dlpfc = Figure2.get_event_subspaces(;subject="ALL", rtime_min=120.0,area="DLPFC")
    fname_cue_fef, fname_mov_fef = Figure2.get_event_subspaces(;subject="ALL", rtime_min=120.0,area="FEF")

    with_theme(PlotUtils.plot_theme) do
        fig = Figure()
        axes = [Axis(fig[1,i]) for i in 1:2]
        for (ax, align,(fname_fef, fname_dlpfc)) in zip(axes, ["cue","mov"],[(fname_cue_fef, fname_cue_dlpfc),(fname_mov_fef, fname_mov_dlpfc)])
            bins, f1score_dlpfc,windows,latencies = h5open(fname_dlpfc) do fid
                read(fid, "bins"), read(fid, "f1score"), read(fid,"window"), read(fid, "latency")
            end
            bins, f1score_fef,windows,latencies = h5open(fname_fef) do fid
                read(fid, "bins"), read(fid, "f1score"), read(fid,"window"), read(fid, "latency")
            end
            idxl = searchsortedfirst(latencies, latency[align],rev=true)
            idxw = searchsortedfirst(windows, window[align])
            
            if plottype == :scatter
                scatter!(ax, f1score_dlpfc[idxw, idxl, 1,:], f1score_fef[idxw,idxl,1,:])
                ax.xlabel = "F₁ score DLPFC"
                ax.ylabel = "F₁ score FEF"
                ablines!(ax, 0.0 ,1.0,color="black")
            else
                # barplot
                nn_dlpfc = size(f1score_dlpfc,4)
                nn_fef = size(f1score_fef,4)
                boxplot!(ax, fill(1.0, nn_fef), f1score_fef[idxw, idxl, 1 ,:],label="FEF", color=PlotUtils.fef_color)
                boxplot!(ax, fill(2.0, nn_dlpfc), f1score_dlpfc[idxw, idxl, 1 ,:], label="DLPFC", color=PlotUtils.dlpfc_color)
                ax.ylabel = "F₁ score"
                ax.xticklabelsvisible = false
                ax.xticksvisible = true
                ax.xticks = [1,2]
                ax.bottomspinevisible = true
                # non-parametric distribution test
                l_fef = percentile(f1score_fef[idxw, idxl, 1, :], 5)
                u_dlpfc = percentile(f1score_dlpfc[idxw, idxl, 1, :], 95)
                @show l_fef u_dlpfc
                @show MannWhitneyUTest(f1score_fef[idxw, idxl, 1, :], f1score_dlpfc[idxw, idxl, 1, :])
            end

            if align == "cue"
                ax.title = "Go-cue onset"
            else
                ax.title = "Movement onset"
            end
        end
        ax = axes[2]
        ax.ylabelvisible = false 
        if plottype != scatter
            axislegend(axes[1])
        end
        fig
    end
end
end #module