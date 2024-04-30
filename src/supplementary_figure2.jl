# Per monkey subspace analysis
using EventOnsetDecoding
using HypothesisTests
using MultivariateStats
using DataProcessingHierarchyTools
using JLD2
using HDF5
using CairoMakie
const DPHT = DataProcessingHierarchyTools

include("figure2.jl")

sessions_j = ["J/20140807/session01", "J/20140828/session01", "J/20140904/session01", "J/20140905/session01"]
sessions_w = ["W/20200106/session02", "W/20200108/session03", "W/20200109/session04", "W/20200113/session01", "W/20200115/session03", "W/20200117/session03", "W/20200120/session01", "W/20200121/session01"]

function plot(;do_save=true,kvs...)
    fig = Figure(resolution=(700,300))
    lgwt = GridLayout()
    fig[1,1] = lgwt
    rowsize!(fig.layout, 1, Relative(0.01))
    Label(lgwt[1,1, Top()], "Monkey W", tellwidth=false, tellheight=true)
    lgw = GridLayout()
    fig[2,1] = lgw
    fname_cue, fname_mov = get_event_subspaces(;subject="W")
    plot_event_onset_subspaces!(lgw, fname_cue, fname_mov;show_colorbar=false, kvs...)
    lgjt = GridLayout()
    fig[1,2] = lgjt
    Label(lgjt[1,1, Top()], "Monkey J", tellwidth=false, tellheight=true)
    lgj = GridLayout()
    fig[2,2] = lgj
    fname_cue, fname_mov = get_event_subspaces(;subject="J")
    plot_event_onset_subspaces!(lgj, fname_cue, fname_mov;ylabelvisible=false, yticklabelsvisible=false, kvs...)

    rowgap!(fig.layout, 1, 0.0)
    fig
end

