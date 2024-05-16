"""
Go-cue onset and movement onset spaces in DLPFC
"""
module FigureS7
include("figure2.jl")

function plot(;do_save=true, kvs...)
    fname_cue, fname_mov = Figure2.get_event_subspaces(;subject="ALL", rtime_min=120.0,area="DLPFC")
    fig = Figure2.plot_event_onset_subspaces(fname_cue, fname_mov;max_latency=70.0, kvs...)
    if do_save
        save(joinpath("figures","manuscript","subspace_dlpfc.pdf"),fig)
    end
    fig
end

end # module
