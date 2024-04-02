using LinearAlgebra
using TimeResolvedDecoding
using MultivariateStats
using HDF5
using JLD2
using Random
using EventOnsetDecoding
using ProgressMeter
using CairoMakie
include("utils.jl")
include("plot_utils.jl")

# get the weights for the go-cue and movement decoders
# train a decoder to decode target location during delay 2
# compare the weights for the two decoders
# color points by preferred location, with perhaps saturation (or something) indicating the strength of tuning

function run(;subject="W", fname="data/ppsth_fef_cue.jld2", window=25.0,nruns=100, kvs...)
    # threshold = 0.5
    #fname_cue, fname_mov = PrefrontalManifoldGeometry.Figure2.get_event_subspaces(;nruns=100,subject="W")
    #weights = h5open(fname_cue) do fid
    #    read(fid, "weights")
    #end
    
    fname_results = replace(fname, "ppsth"=>"target_information_$(subject)")
    if !isfile(fname_results)
        ppstht, labelst, rtimest,trialidxt = JLD2.load(fname, "ppsth", "labels","rtimes","trialidx")
        nbins = size(ppstht.counts,1)
        # get data for each monkey
        allsessions = Utils.DPHT.get_level_path.("session", ppstht.cellnames)
        subjects = Utils.DPHT.get_level_name.("subject", ppstht.cellnames)
        sessions = unique(allsessions[subjects.==subject])
        nt = fill(0, length(sessions))
        ncells = fill(0, length(sessions))
        Xa = fill(0.0, size(ppstht.counts)) 
        labels = Vector{Vector{Int64}}(undef, sum(subjects.==subject))
        offset = 0
        for (sessionidx,session) in enumerate(sessions)
            X, _labels, _rtimes = Utils.get_session_data(session, ppstht, trialidxt, labelst, rtimest;mean_subtract=false, variance_stabilize=false,kvs...)
            nt[sessionidx] = size(X,2)
            _nt = nt[sessionidx]
            ncells[sessionidx] = size(X,3)
            _ncells = ncells[sessionidx]
            Xa[:,1:_nt,offset+1:offset+_ncells] .= X
            ulabels = unique(_labels)
            sort!(ulabels)
            _labels = [searchsortedfirst(ulabels, l) for l in _labels]
            labels[offset+1:offset+_ncells] .= fill(_labels, _ncells)
            offset += _ncells
        end
        Xa = Xa[:,1:maximum(nt),1:offset]
        X2,bins2 = Utils.rebin2(Xa, ppstht.bins, window)
        X2 = X2[1:length(bins2),:,:]
        nbins = length(bins2)
        # training-testing pseudo-population loop
        decoder = SubspaceLDA
        decoder_results = Vector{TimeResolvedDecoding.TimeResolvedDecoder{TimeResolvedDecoding.IdentityTransform, decoder}}(undef, nruns)
        @showprogress for i in 1:nruns
            Yt, train_label, test_label = EventOnsetDecoding.sample_trials(permutedims(X2, [3,2,1]), labels; RNG=Random.default_rng(), ntrain=1500, ntest=300)
            Yt = permutedims(Yt, [3,1,2])
            trainidx = [[1:length(train_label);]]
            decoder_results[i] = TimeResolvedDecoding.decode_target_location(I(size(Yt,1)), Yt, [train_label;test_label],trainidx, [[i] for i in 1:size(Yt,2)],decoder=decoder)
        end
        JLD2.save(fname_results, Dict("decoder_results"=>decoder_results, "bins"=>bins2,"window"=>window))
    end
    decoder_results, bins2 = JLD2.load(fname_results, "decoder_results","bins")
    decoder_results, bins2
end

function get_triu_entries(X::Matrix{T}) where T <: Real
    n1,n2 = size(X)
    n1 == n2 || error("Matrix should be square")
    n = div(n1*(n1-1),2)
    Xu = zeros(T, n)
    k = 1
    for i in 1:n1-1
        for j in i+1:n1
            Xu[k] = X[j,i]
            k += 1
        end
    end
    Xu
end

function get_weight_correlation(decoder_results,idx0::Int64)
    nruns = length(decoder_results)
    D = fill(0.0, nruns,nruns)
    for i in 1:nruns
        Pi = projection(decoder_results[i].decoder[idx0]) 
        # find an orthonormal basis
        qi,_ = qr(Pi)
        for j in 1:nruns
            Pj = projection(decoder_results[j].decoder[idx0]) 
            qj,_ = qr(Pj)
            D[j,i] = maximum(qi[:,1:3]'*qj[:,1:3])
        end
    end
    D
end

function get_weight_correlation(decoder_results,weights::Matrix{T}, idx0::Int64) where T <: Real
    n1 = length(decoder_results)
    n2 = size(weights,2)
    D = fill(0.0, n2,n1)
    for i in 1:n1
        Pi = projection(decoder_results[i].decoder[idx0]) 
        # find an orthonormal basis
        qi,_ = qr(Pi)
        for j in 1:n2
            w = weights[:,j]
            w ./= norm(w)
            D[j,i] = maximum(qi[:,1:3]'w)
        end
    end
    D
end

function get_all_results()
    results = Dict()
    results["mov"] = Dict()
    results["cue"] = Dict()
    results["mov"]["J"], bins_mov = run(;subject="J", fname="data/ppsth_fef_mov.jld2", mean_subtract=false, variance_stabilize=false)
    results["mov"]["W"], _ = run(;subject="W", fname="data/ppsth_fef_mov.jld2", mean_subtract=false, variance_stabilize=false)

    results["cue"]["J"], bins_cue = run(;subject="J", fname="data/ppsth_fef_cue.jld2", mean_subtract=false, variance_stabilize=false)
    results["cue"]["W"], _ = run(;subject="W", fname="data/ppsth_fef_cue.jld2", mean_subtract=false, variance_stabilize=false)
    results, bins_cue, bins_mov
end

function get_correlation_results(;latency::Dict=Dict("mov"=>0.0, "cue"=>40.0), window::Dict=Dict("mov"=>35.0, "cue"=>15.0))
    results,bins_cue, bins_mov = get_all_results()
    D = Dict()
    D["mov"] = Dict()
    D["mov"]["J"] = Dict()
    D["mov"]["W"] = Dict()
    D["cue"] = Dict()
    D["cue"]["J"] = Dict()
    D["cue"]["W"] = Dict()

    for subject in ["J","W"]
        fname_cue, fname_mov = PrefrontalManifoldGeometry.Figure2.get_event_subspaces(;nruns=100,subject=subject)
        for (fname,align,bins) in zip([fname_cue, fname_mov],["cue","mov"],[bins_cue, bins_mov])
            weights, _latency, _window = h5open(fname) do fid
                read(fid, "weights"), read(fid, "latency"), read(fid, "window")
            end
            idxl = searchsortedfirst(_latency, latency[align], rev=true)
            idxw = searchsortedfirst(_window, window[align])
            w = weights[:,1,idxw,idxl,1,:]
            if align == "mov"
                idx0 = searchsortedfirst(bins, -latency["mov"]-window["mov"])
            else
                idx0 = searchsortedfirst(bins, latency["mov"])
            end
            D2 = get_weight_correlation(results[align][subject], w, idx0)
            DD = get_weight_correlation(results[align][subject], idx0)
            D[align][subject]["DX"] = D2
            D[align][subject]["DD"] = get_triu_entries(DD) 
            D[align][subject]["idx0"] = idx0
            D[align][subject]["bins"] = bins 
        end
    end
    D
end

function create_weight_correlation_figure(;kvs...)
    D = get_correlation_results(;kvs...)
    create_weight_correlation_figure(D)
end
    
function create_weight_correlation_figure(D)
    
    with_theme(PlotUtils.plot_theme) do
        fig = Figure()
        axes = [Axis(fig[i,j]) for i in 1:2, j in 1:2]
        for (jj,subject) in enumerate(["J","W"])
            for (ii,align) in enumerate(["cue","mov"])
                DX = D[align][subject]["DX"]
                DD = D[align][subject]["DD"]
                ax = axes[jj,ii]
                x = [fill(1.0, length(DX));fill(2.0, length(DD))]
                boxplot!(ax, fill(1.0, length(DX)), DX[:];width=0.5,label="Target↔Subspace")
                boxplot!(ax, fill(2.0, length(DD)), DD[:];width=0.5, label="Target↔Target")
                ax.ylabel = "Correlation $subject"
                ax.xlabel = align 
                ax.xticklabelsvisible = false
                ax.xticksvisible = false
                ax.bottomspinevisible = false
            end
        end
        linkyaxes!(axes[1,:]...,)
        linkyaxes!(axes[2,:]...,)
        for ax in axes[:,2]
            ax.yticklabelsvisible = false
            ax.yticksvisible = false
            ax.leftspinevisible = false
            ax.ylabelvisible = false
        end
        for ax in axes[1,:]
            ax.xticklabelsvisible = false
            ax.xlabelvisible = false
        end
        # legend
        axislegend(axes[1,1],halign=:left)
        fig
    end

end

function create_decoding_figure()
    # plot target decoding performance for both monkeys, aligned to both go-cue and movement onset

    results = get_all_results()
    with_theme(PlotUtils.plot_theme) do
        fig = Figure()
        axes = [Axis(fig[i,j]) for i in 1:2, j in 1:2] 
        for (ii,(align,bins)) in enumerate(zip(["cue","mov"], [bins_cue, bins_mov]))
            for (jj,subject) in enumerate(["J","W"])
                perf = cat([results[align][subject][i].perf[:,1,1] for i in 1:length(results_mov)]..., dims=2)
                μ = dropdims(mean(perf,dims=2),dims=2)
                σ = dropdims(std(perf,dims=2),dims=2)
                ax = axes[jj,ii]
                vlines!(ax, 0.0, color=:black, linestyle=:dot)
                if subject == "W"
                    hlines!(ax, 0.25, color=:black, linestyle=:dot)
                else
                    hlines!(ax, 1/7, color=:black, linestyle=:dot)
                end
                lines!(ax, bins, μ)
                band!(ax, bins, μ-2σ, μ+2σ)
                ax.ylabel = "Performance $(subject)"
            end
        end
        for ax in axes[1,:]
            ax.xticklabelsvisible = false
        end
        for ax in axes[:,2]
            ax.yticklabelsvisible = false
            ax.ylabelvisible = false
        end
        linkyaxes!(axes[1,:]...)
        linkyaxes!(axes[2,:]...)
        axes[2,1].xlabel = "Time from go-cue"
        axes[2,2].xlabel = "Time from movement"
        fig
    end
end