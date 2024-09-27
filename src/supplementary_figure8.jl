module FigureS8
using MultivariateStats
using CRC32c
using Makie
using CairoMakie
using Colors
using JLD2


using ..Utils
using ..PlotUtils

include("fa.jl")

function plot()
    fname = joinpath("data","supfig5.data.jld2")
    if isfile(fname)
        @show fname
        r²j, r²w, Y, lrt, d_shared = JLD2.load(fname, "r²j", "r²w", "Y", "lrt", "d_shared")
    else
        reg_window = (0.0, 50.0)
        # finding the number of factors
        d_shared = let
            h = crc32c(string(reg_window))
            q = string(h, base=16)
            fname = joinpath("data","james_whiskey_shared_dimensionality_$q.jld2")
            if isfile(fname)
                d_shared = JLD2.load(fname, "d_shared")
            else
                sessionsidx = Dict("james" => 1:4, "whiskey" => 1:8)
                locations = Dict("james" => [1,2,3,4,6,7,8], "whiskey" => 1:4)
                d_shared = Vector{Union{Missing, Int64}}(undef, 0)
                for subject in ["whiskey","james"]
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
        subject="whiskey"
        sessionidx = 4
        location=2

        ppsth,tlabels,_trialidx, rtimes = load_data(nothing;area="fef", raw=true);
        sessions = unique(Utils.DPHT.get_level_path.("session", ppsth.cellnames))
        subjects = Utils.DPHT.get_level_name.("subject", sessions) 
        sidx = findall(subjects.==subject)
        session = sessions[sidx[4]]
        # get population responses for a single session
        X, bins, labels, rtimes = get_session_data(session, ppsth, tlabels, _trialidx, rtimes;rtmin=120.0, rtmax=300.0, mean_subtract=true, variance_stabilize=true)

        # get a window in the late delay 2 period
        Y, lrt, β, r², pv = get_rtime_regression(X, bins, rtimes, labels,location;reg_window=reg_window)

        # summary
        r²w, nsw = MovementDecoders.get_rtime_r²("whiskey",:cue,reg_window=reg_window)
        r²ws, _ = MovementDecoders.get_rtime_r²("whiskey",:cue;reg_window=reg_window, do_shuffle=true)

        r²j, nsj = MovementDecoders.get_rtime_r²("james",:cue, reg_window=reg_window)
        r²js, _ = MovementDecoders.get_rtime_r²("james",:cue;reg_window=reg_window, do_shuffle=true)
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
        fname = joinpath("figures","manuscript","khanna_analysis.pdf")
        save(fname, fig)
	    fig
    end
end
end #module