module FigureS8
using MultivariateStats
using CRC32c
using Makie
using CairoMakie
using Colors
using JLD2
using StatsBase
using LinearAlgebra
using Distributions

using ..Utils
using ..PlotUtils
using ..Regression

include("fa.jl")

function get_rtime_r²(subject::String, alignment::Symbol;reg_window=(-400.0, -50.0), rtmin=120.0, rtmax=300.0, do_shuffle=false, redo=false, kvs...)
    h = crc32c(string(alignment))
    if reg_window != (-400.0, -50.0)
        h = crc32c(string(reg_window), h)
    end
    q = string(h, base=16)
    if do_shuffle
        fname = "$(subject)_fa_regression_summary_data_$q.jld2"
    else
        fname = "$(subject)_fa_regression_summary_shuffle_data_$q.jld2"
    end
    if isfile(fname) && !redo
        r²,skipped = JLD2.load(fname, "r²","skipped")
    else
        if lowercase(subject) == "j"
            sessions = Utils.sessions_j
        elseif lowercase(subject) == "w"
            sessions = Utils.sessions_w
        else
            error("Unkown subject $(subject)")
        end
        data_fname = joinpath(@__DIR__, "..", "data", "ppsth_fef_$(alignment)_raw.jld2") 
        ppstht, tlabelst, rtimest, trialidxt = JLD2.load(data_fname, "ppsth","labels", "rtimes", "trialidx")
        r² = Float64[]
        skipped = 0
        for session in sessions
            X, bins, labels, rtimes = Utils.get_session_data(session, ppstht, trialidxt, tlabelst, rtimest;rtmin=rtmin, rtmax=rtmax, mean_subtract=true, variance_stabilize=true)
            bidx = searchsortedfirst(bins, reg_window[1]):searchsortedlast(bins, reg_window[2])
            for location in locations[uppercase(subject)]
                tidx = labels.==location
                Xl = dropdims(mean(X[bidx,tidx,:],dims=1),dims=1)
                Σ = cov(Xl, dims=1)
                σ² = diag(Σ)
                try
                    pp = StatsBase.fit(Gamma, σ²)
                
                    ppq = quantile(pp, 0.01)
                    cidx = findall(σ² .> ppq)
                    Xlc = permutedims(Xl[:, cidx],[2,1])
                    #@debug "Cells" olength=size(Xl,2) klength=length(cidx)
                    fa = StatsBase.fit(MultivariateStats.FactorAnalysis, Xlc;maxoutdim=1, method=:em)
                    Y = permutedims(MultivariateStats.predict(fa, Xlc), [2,1])
                    Y .-= mean(Y)
                    _lrt = log.(rtimes[tidx])
                    _lrt .-= mean(_lrt)
                    if do_shuffle
                        shuffle!(_lrt)
                    end
                    β,_r²,pv = MovementDecoders.llsq_stats(Y, _lrt)
                    push!(r², _r²)
                catch ee
                    skipped += 1
                end
            end
        end
        JLD2.save(fname, Dict("alignment" => alignment, "reg_window"=>reg_window, 
                              "rtmin"=>rtmin, "rtmax"=>rtmax, "do_shuffle" => do_shuffle,
                              "r²"=>r², "skipped"=>skipped))
    end
	r², skipped
end

function get_rtime_regression(X::Array{T,3}, bins::AbstractArray{Float64},rtime::Vector{Float64}, location_labels::Vector{Int64}, location::Int64;reg_window=(-400.0, -50.0), do_shuffle=false,kvs...) where T <: Real
    bidx = searchsortedfirst(bins, reg_window[1]):searchsortedlast(bins, reg_window[2])
    tidx = location_labels.==location
    Xl = dropdims(mean(X[bidx,tidx,:],dims=1),dims=1)
    Σ = cov(Xl, dims=1)
    σ² = diag(Σ)
    pp = StatsBase.fit(Gamma, σ²)

    ppq = quantile(pp, 0.01)
    cidx = findall(σ² .> ppq)
    @show size(Xl) length(cidx)
    Xlc = permutedims(Xl[:, cidx],[2,1])
    fa = StatsBase.fit(MultivariateStats.FactorAnalysis, Xlc;maxoutdim=1, method=:em)
    @show fa
    Y = permutedims(MultivariateStats.predict(fa, Xlc), [2,1])
    @show Y
    Y .-= mean(Y)
    _lrt = log.(rtime[tidx])
    _lrt .-= mean(_lrt)
    if do_shuffle
        shuffle!(_lrt)
    end
    β,_r²,pv = Regression.llsq_stats(Y, _lrt)
    Y, _lrt, β, _r², pv
end


function plot()
    fname = joinpath(@__DIR__, "..", "data","supfig5.data.jld2")
    if isfile(fname)
        r²j, r²w, Y, lrt, d_shared,β = JLD2.load(fname, "r²j", "r²w", "Y", "lrt", "d_shared","β")
    else
        reg_window = (0.0, 50.0)
        # finding the number of factors
        d_shared = let
            h = crc32c(string(reg_window))
            q = string(h, base=16)
            fname = joinpath(@__DIR__, "..", "data","j_w_shared_dimensionality_$q.jld2")
            @show fname
            if isfile(fname)
                d_shared = JLD2.load(fname, "d_shared")
            else
                d_shared = Vector{Union{Missing, Int64}}(undef, 0)
                for subject in ["j","w"]
                    for sessionidx in sessionsidx[subject]
                        for location in locations[subject]
                            try
                                ll, d, ii = get_shared_dimensionality(subject, sessionidx, location, method=:em, reg_window=reg_window)
                                push!(d_shared, d)
                            catch ee
                                continue
                            end
                        end
                    end
                end
                JLD2.save(fname, Dict("d_shared"=>d_shared, "reg_window"=>reg_window))
            end
            d_shared
        end

        d_shared = filter(!ismissing, d_shared) 
        counts = countmap(d_shared)
        x_values = sort(collect(keys(counts)))
        subject="W"
        sessionidx = 4
        session = Utils.sessions_w[4]
        location=2

        ppsth,tlabels,_trialidx, rtimes = load_data(nothing;area="fef", raw=true);
        #sessions = unique(Utils.DPHT.get_level_path.("session", ppsth.cellnames))
        #subjects = Utils.DPHT.get_level_name.("subject", sessions) 
        #sidx = findall(subjects.==subject)
        #session = sessions[sidx[4]]
        # get population responses for a single session
        X, bins, labels, rtimes = get_session_data(session, ppsth, _trialidx, tlabels, rtimes;rtmin=120.0, rtmax=300.0, mean_subtract=true, variance_stabilize=true)

        # get a window in the late delay 2 period
        @show reg_window
        Y, lrt, β, r², pv = get_rtime_regression(X, bins, rtimes, labels,location;reg_window=reg_window)

        # summary
        r²w, nsw = get_rtime_r²("w",:cue,reg_window=reg_window)
        r²ws, _ = get_rtime_r²("w",:cue;reg_window=reg_window, do_shuffle=true)

        r²j, nsj = get_rtime_r²("j",:cue, reg_window=reg_window)
        r²js, _ = get_rtime_r²("j",:cue;reg_window=reg_window, do_shuffle=true)
        xw = 2*rand(length(r²w)) .- 1.0
        xj = 2*rand(length(r²j)) .- 1.0

        show_cdf = false
        d_shared = filter(!ismissing, d_shared) 
        counts = countmap(d_shared)
        x_values = sort(collect(keys(counts)))
        JLD2.save(fname, Dict("r²w"=>r²w, "r²j"=>r²j, "d_shared"=>d_shared,"Y"=>Y, "lrt"=>lrt,"β"=>β))
    end
    with_theme(plot_theme) do
        fig = Figure(resolution=(400,250))
        ax = Axis(fig[1,2])
        scatter!(ax, Y[:], lrt, markersize=5px)
        ablines!(ax, [β[end]], [β[1]],linestyle=:dot, color="black")
        Label(fig[1,2, Top()], L"r^2 = %$(round(r², sigdigits=2)), p=%$(round(pv, sigdigits=2))")
        ax.xlabel = "1st factor projection"
        ax.ylabel = "log(rtime)"

        # new layout
        lg = GridLayout()
        fig[1,1] = lg
        # summary CDF
        if show_cdf
            ax1 = Axis(lg[1,1])
            x = sort([r²w;r²j])
            y = range(0.0, stop=1.0, length=length(x))
            xs = sort([r²ws;r²js])
            q = HypothesisTests.SignedRankTest(x,xs)
            @show q
            lines!(ax1, x, y, color="black", label="data")
            lines!(ax1, xs, y, color=RGB(0.8, 0.8, 0.8), label="shuffled")
            axislegend(ax1, halign=:right, valign=:bottom)
            ax1.bottomspinevisible = false
            ax1.xticksvisible = false
            ax1.xticklabelsvisible = false

            # summary points
            ax2 = Axis(lg[2,1])
            linkxaxes!(ax1, ax2)
            scatter!(ax2, r²w, xw, markersize=5px)
            scatter!(ax2, r²j, xj, markersize=5px)
            ax2.leftspinevisible = false
            ax2.yticksvisible = false
            ax2.yticklabelsvisible = false
            ax2.xlabel = L"r^2"
            rowsize!(lg, 2, Relative(0.2))
        else
            ax1 = Axis(lg[1,1])
            barplot!(ax1, x_values, [counts[k] for k in x_values])
            ax1.xticks = x_values
            ax1.xlabel = L"d_{\mathrm{shared}}"
            ax1.ylabel = "count"
        end
        colsize!(fig.layout, 1, Relative(0.3))
        labels = [Label(fig[1,1, TopLeft()], "A"),
                Label(fig[1,2, TopLeft()], "B")]
        fname = joinpath(@__DIR__, "..", "figures","manuscript","khanna_analysis.pdf")
        save(fname, fig)
	    fig
    end
end
end #module