module Figure3
using Makie
using CairoMakie
using JLD2
using CRC32c

#include("utils.jl")
include("regression.jl")
include("trajectories.jl")
#include("plot_utils.jl")

"""
Data for the figure comparing path length, average speed and post-cue period in terms of
how much reaction time variance they explain
"""
function get_reaction_time_regressors(;t0=85.0, t1=35.0, redo=false, do_save=true,area="FEF", do_shuffle=false,rtmin=120.0)
    # TODO: Check why the r² values went down. I think it was after I changed from summing up the squares
    # .     to summing about the square roots.
    h = "" 
    q = zero(UInt32)
    if t0 != 85.0 
        q = crc32c(string(t0),q)
    end
    if t1 != 35.0
        q = crc32c(string(t1),q)
    end
    if rtmin != 120.0
        q = crc32c("rtmin=$(rtmin)",q)
    end
    if q > 0
        h = "_$(string(q, base=16))"
    else
        h = ""
    end
    if area == :FEF
        fname = "reaction_time_predictors_new$(h).jld2"
    else
        fname = "reaction_time_predictors_new_$(area)$(h).jld2"
    end
    if isfile(fname) && !redo
        stats_results = JLD2.load(fname, "stats_results")
    else
        stats_results = Dict{String,Any}()
        stats_results["params"] = Dict("t0"=>t0, "t1"=>t1)
        stats_results["path_length"] = Dict{String,Any}()
        stats_results["path_length2"] = Dict{String,Any}()
        stats_results["avg_speed"] = Dict{String,Any}()
        stats_results["delay2"] = Dict{String,Any}()
        stats_results["postcue"] = Dict{String,Any}()
        stats_results["path_length"]["W"] = Dict{String,Any}(zip(["S","lrt","tridx"], get_path_length_and_rtime("W", t0, t1;area=area,do_shuffle=do_shuffle,rtmin=rtmin)))
        stats_results["path_length"]["J"] = Dict{String,Any}(zip(["S","lrt", "tridx"], get_path_length_and_rtime("J", t0, t1;area=area, do_shuffle=do_shuffle, rtmin=rtmin)))
        stats_results["path_length2"]["W"] = Dict{String,Any}(zip(["S","lrt","tridx"], get_path_length_and_rtime("W", t0, t1, operation=:path_length2,area=area, do_shuffle=do_shuffle, rtmin=rtmin)))
        stats_results["path_length2"]["J"] = Dict{String,Any}(zip(["S","lrt", "tridx"], get_path_length_and_rtime("J", t0, t1, operation=:path_length2, area=area, do_shuffle=do_shuffle, rtmin=rtmin)))
        stats_results["avg_speed"]["W"] = Dict{String,Any}(zip(["S", "lrt", "tridx"], get_path_length_and_rtime("W", t0, t1,operation=:mean_speed, area=area, do_shuffle=do_shuffle,rtmin=rtmin)))
        stats_results["avg_speed"]["J"] = Dict{String,Any}(zip(["S","lrt", "tridx"], get_path_length_and_rtime("J", t0, t1,operation=:mean_speed, area=area, do_shuffle=do_shuffle,rtmin=rtmin)))
        stats_results["delay2"]["W"] = Dict{String,Any}(zip(["lrt","S", "tridx"], explain_rtime_variance("W", :cue;realign=true, area=area,rtmin=rtmin)))
        stats_results["delay2"]["J"] = Dict{String,Any}(zip(["lrt","S", "tridx"], explain_rtime_variance("J", :cue;realign=true,area=area,rtmin=rtmin)))
        stats_results["postcue"]["W"] = Dict{String,Any}(zip(["lrt","S","tridx"], explain_rtime_variance("W", :cue;realign=true, reg_window=(0.0, 50.0), area=area,rtmin=rtmin)))
        #TODO: Why are so many trials excluded here?
        stats_results["postcue"]["J"] = Dict{String,Any}(zip(["lrt","S", "tridx"],explain_rtime_variance("J", :cue;realign=true, reg_window=(0.0, 50.0), area=area,rtmin=rtmin)))
      
        if any(isnan.(stats_results["path_length"]["W"]["S"]))
            @warn "NaN encountered"
        end
        # make sure we are using the same indices
        k1 = "postcue"
        k2 = "path_length"
        tridxj = intersect(stats_results[k1]["J"]["tridx"],
                stats_results[k2]["J"]["tridx"])
        tridxw = intersect(stats_results[k1]["W"]["tridx"],
                stats_results[k2]["W"]["tridx"])
        tridxpcj = findall(in(tridxj), stats_results[k1]["J"]["tridx"])
        tridxplj = findall(in(tridxj), stats_results[k2]["J"]["tridx"])
        tridxpcw = findall(in(tridxw), stats_results[k1]["W"]["tridx"])
        tridxplw = findall(in(tridxw), stats_results[k2]["W"]["tridx"])

        for (kk,_tridxjw) in zip(["path_length","path_length2","avg_speed","postcue"],[(tridxplj,tridxplw),(tridxplj,tridxplw),(tridxplj,tridxplw),(tridxpcj, tridxpcw)])
            for (subject,_tridx) in zip(["J","W"], _tridxjw) 
                @debug subject kk
                xj = stats_results[kk][subject]
                @debug size(xj["S"]) size(xj["lrt"]) size(xj["tridx"])
                if size(xj["S"]) == (0,)
                    error("No trials found. subject=$(subject) kk=$(kk)")
                end
                xj["S"] = xj["S"][_tridx]
                xj["lrt"] = xj["lrt"][_tridx]
                xj["tridx"] = xj["tridx"][_tridx]
            end
        end
        
        for (k,v) in stats_results
            if k == "params"
                continue
            end
            xj = v["J"]
            xw = v["W"]
            lrt = [xw["lrt"];xj["lrt"]]
            S = cat(xw["S"],xj["S"],dims=1)
            idx = isfinite.(S)
            S = repeat(S[idx],1,1)
            v["both"] = Dict{String,Any}()
            vb = v["both"]
            @show k size(S) size(lrt[idx])
            vb["β"], vb["r²"], vb["pv"], vb["rss"] = llsq_stats(S, lrt[idx])
            vb["lrt"], vb["S"] = (lrt[idx], S)
            # regress individual monkeys too
            idxj = isfinite.(xj["S"])
            xj["β"], xj["r²"], xj["pv"], xj["rss"] = llsq_stats(repeat(xj["S"][idxj],1,1), xj["lrt"][idxj])
            idxw = isfinite.(xw["S"])
            xw["β"], xw["r²"], xw["pv"], xw["rss"] = llsq_stats(repeat(xw["S"][idxw],1,1), xw["lrt"][idxw])
            # for nested regression, simply add one variable to the other
        end
        # now do hierarchial regression
        for (k1, k2) in [("postcue","path_length"),("path_length","postcue"),("postcue","path_length2"),("path_length2","postcue"), ("avg_speed","postcue"), ("postcue","avg_speed"), ("avg_speed", "path_length2"),("path_length2", "avg_speed"),("avg_speed","path_length"),("postcue", "avg_speed")]
            kk = "$(k1)→$(k2)"
            stats_results[kk] = Dict{String,Any}()
            for subject in ["J","W"]
                stats_results[kk][subject] = Dict{String, Any}()
                qq = stats_results[kk][subject]
                Splw = stats_results[k2][subject]["S"]
                Spcw = stats_results[k1][subject]["S"]
                lrtplw = stats_results[k2][subject]["lrt"]
                lrtpcw = stats_results[k1][subject]["lrt"]
                @assert lrtplw == lrtpcw
                S = [Spcw Splw]
                lrt = lrtpcw
                qqidx = isfinite.(dropdims(sum(S,dims=2),dims=2))
                # residual
                _β = stats_results[k2][subject]["β"]
                rr = Splw*_β[1:end-1]' .+ _β[end] - lrtpcw
                βr, r²r, pvr, rssr = llsq_stats(repeat(Spcw[qqidx],1,1), rr[qqidx])
                qq["r²_res"] = r²r
                β2,r², pv,rss2 = llsq_stats(S[qqidx,:], lrt[qqidx])
                p2 = length(β2)
                β1,_,_,rss1 =  llsq_stats(S[qqidx,1:1], lrt[qqidx])
                p1 = length(β1)
                n = length(lrt)
                F = (rss1 - rss2)/(p2-p1)
                F /= rss2/(n-p2)
                pv = 1.0 - cdf(FDist(p2-p1, n-p2), F)
                qq["β"] = β2
                qq["β1"] = β1
                qq["rss"] = rss2
                qq["rss1"] = rss1
                qq["pv"] = pv
                qq["F"] = F
                qq["r²"] = r²
            end
            # both
            stats_results[kk]["both"] = Dict{String, Any}()
            qq = stats_results[kk]["both"]
            Splw = stats_results[k2]["W"]["S"]
            Splj = stats_results[k2]["J"]["S"]
            Spcw = stats_results[k1]["W"]["S"]
            Spcj = stats_results[k1]["J"]["S"]

            lrtplw = stats_results[k2]["W"]["lrt"]
            lrtplj = stats_results[k2]["J"]["lrt"]
            lrtpcw = stats_results[k1]["W"]["lrt"]
            lrtpcj = stats_results[k1]["J"]["lrt"]

            @assert lrtplj == lrtpcj
            S = [[Spcw;Spcj] [Splw;Splj]]
            lrt = [lrtplw;lrtplj]
            qqidx = isfinite.(dropdims(sum(S,dims=2),dims=2))
            β2,r², pv,rss2 = llsq_stats(S[qqidx,:], lrt[qqidx])
            p2 = length(β2)
            β1,_,_,rss1 =  llsq_stats(S[qqidx,1:1], lrt[qqidx])
            p1 = length(β1)
            n = length(lrt)
            F = (rss1 - rss2)/(p2-p1)
            F /= rss2/(n-p2)
            pv = 1.0 - cdf(FDist(p2-p1, n-p2), F)
            qq["β"] = β2
            qq["β1"] = β1
            qq["rss"] = rss2
            qq["rss1"] = rss1
            qq["pv"] = pv
        end
        #now do residual

        stats_results["path_length"]["legend_anchor"] = (0.45, 0.85)
        stats_results["path_length2"]["legend_anchor"] = (0.45, 0.85)
        stats_results["avg_speed"]["legend_anchor"] = (0.40, 0.85)
        stats_results["delay2"]["legend_anchor"] = (0.45, 0.85)
        stats_results["postcue"]["legend_anchor"] = (0.45, 0.85)
        if do_save
            JLD2.save(fname, Dict("stats_results"=>stats_results))
        end
    end
    stats_results
end

"""
Plot trajectory illustration using the supplied axis `ax`
"""
function plot_trajectory_illustration!(lg)
    nn = 500
    θ = range(0.0, stop=π, length=nn)
    a = 1.0
    b1 = 1.5
    b2 = 1.1
    traj1 = [a*cos.(θ) b1*sin.(θ)]
    traj2 = [a*cos.(θ) b2*sin.(θ)]
    traj3 = traj2 .+ [0.0 0.1]
    dm,idx1 = compute_triangular_path_length(traj1)
    dm,idx2 = compute_triangular_path_length(traj2)
    tidx1 = [1, idx1, idx1, nn]
    tidx2 = [1, idx2, idx2, nn]
    ss = 40 
    l2 = sqrt.(sum(abs2, diff(traj2,dims=1),dims=2))
    delta = sum(l2[1:ss])
    with_theme(theme_minimal()) do
        ax = Axis(lg[1,1], autolimitaspect=1,alignmode=Inside())
        ax2 = Axis(lg[1,3], autolimitaspect=1, alignmode=Inside())
        linkaxes!(ax, ax2)
        lines!(ax, traj1[:,1], traj1[:,2],color="red")
        linesegments!(ax, traj1[tidx1,1], traj1[tidx1,2],color="red", linestyle=:dot)
        plot_normals!(ax, traj1;ss=(traj,i)->advance(traj,i,delta), linelength=0.16,color="red")
        lines!(ax, traj2[:,1], traj2[:,2],color="blue")
        linesegments!(ax, traj2[tidx2,1], traj2[tidx2,2],color="blue",linestyle=:dot)
        plot_normals!(ax, traj2;ss=(traj,i)->advance(traj,i,delta), linelength=0.16,color="blue")
    
        lines!(ax2, traj2[:,1], traj2[:,2],color="blue", label="Short RT")
        plot_normals!(ax2, traj2;ss=(traj,i)->advance(traj,i,delta), linelength=0.16,color="blue")
    
        lines!(ax2, traj3[:,1], traj3[:,2],color="red", label="Long RT")
        plot_normals!(ax2, traj3;ss=(traj,i)->advance(traj,i,delta/2.0), linelength=0.16,color="red")
        scatter!(ax2, traj2[[1,nn],1], traj2[[1,nn],2], marker=:rect, markersize=20px, color=[RGB(0.5, 1.0, 1.0),RGB(0.5, 1.0, 0.5)])
        scatter!(ax, traj2[[1,nn],1], traj2[[1,nn],2], marker=:rect, markersize=20px, color=[RGB(0.5, 1.0, 1.0),RGB(0.5, 1.0, 0.5)])
        ll = Legend(lg[1,2], ax2, valign=:top,framevisible=true, padding=5.0, labelsize=12)

        #text!(ax, 0.5, 0.4, text="Same speed\nDifferent length", space=:relative, align=(:center, :top), fontsize=12)
        #text!(ax2, 0.5, 0.4, text="Same length\nDifferent speed", space=:relative, align=(:center, :top), fontsize=12)
        ax.xlabel = "Same speed\nDifferent length" 
        ax.xlabelsize = 12
        ax2.xlabel = "Same length\nDifferent speed"
        ax2.xlabelsize = 12
        colgap!(lg,1,0.1)
        colgap!(lg,2,0.1)
        for _ax in [ax, ax2]
            _ax.xticks = (traj2[[nn,1],1],["Go-cue","Threshold"])
            _ax.yticksvisible = false
            _ax.xticklabelsize=12
            _ax.yticklabelsvisible = false
            _ax.bottomspinevisible = false
            _ax.leftspinevisible = false
        end
    end 
end

function plot(;width=800, height=600, kvs...)
    stats_results = get_reaction_time_regressors(;kvs...)
    qq1 = stats_results["postcue→path_length2"]["J"]
    qq2 = stats_results["path_length2→postcue"]["J"]

    qq1 = stats_results["postcue→path_length2"]["W"]
    qq2 = stats_results["path_length2→postcue"]["W"]
    # tweak to make sure we can see the annotations
    # tweak to make sure we can see the annotations
    ymin = -0.3
    ymax = 0.8
    with_theme(plot_theme) do
        markersize=5px
        fig = Figure(resolution=(width, height))
        lg2 = GridLayout()
        fig[2,1:2] = lg2
        axes = [Axis(lg2[1,i]) for i in 1:3]
        #linkyaxes!(axes...)
        for (ax, vv) in zip(axes, ["postcue", "path_length2","avg_speed"])
            _data = stats_results[vv]
            r² = _data["both"]["r²"]
            β = _data["both"]["β"]
            pv = _data["both"]["pv"]
            r²j = _data["J"]["r²"]
            pvj = _data["J"]["pv"]
            βj = _data["J"]["β"]
            r²w = _data["W"]["r²"]
            pvw = _data["W"]["pv"]
            βw = _data["W"]["β"]
            lstring = [L"$r^2 = %$(round(_r², sigdigits=2))$, $p = $%$(round(_pv,sigdigits=2))" for (_r², _pv) in zip([r², r²w, r²j],[pv, pvw, pvj])]
            S = stats_results[vv]["both"]["S"]
            lrt = stats_results[vv]["both"]["lrt"]
            Sw = stats_results[vv]["W"]["S"]
            Sj = stats_results[vv]["J"]["S"]
            scatter_color = fill(ax.palette.color[][1], length(S))
            scatter_color[length(Sw)+1:end] .= ax.palette.color[][2]
            scatter!(ax, S[:], lrt, color=scatter_color, markersize=markersize)
            if "legend_anchor" in keys(stats_results[vv])
                _anchor = stats_results[vv]["legend_anchor"]
            else
                _anchor = (0.40, 0.85)	
            end
            text!(ax, _anchor[1]-0.04, _anchor[2]+0.1, text=lstring[1], space=:relative, fontsize=12)
            text!(ax, _anchor[1]-0.04, _anchor[2]+0.05, text=lstring[2], space=:relative, color=ax.palette.color[][1], fontsize=12)
            text!(ax, _anchor[1]-0.04, _anchor[2], text=lstring[3], space=:relative, color=ax.palette.color[][2], fontsize=12)
            ablines!(ax, β[2:2], β[1:1], color="black", linestyle=:dot)
            ablines!(ax, βw[2:2], βw[1:1], color=Cycled(1), linestyle=:dot)
            ablines!(ax, βj[2:2], βj[1:1], color=Cycled(2), linestyle=:dot)
            ylims!(ax, ymin, ymax)
        end
        # joint vs single regression
        lg7 = GridLayout()
        fig[1,2] = lg7
        axesp = [Axis(lg7[1,i]) for i in 1:2]
        linkyaxes!(axesp...)
        for (ii,(subject,ax)) in enumerate(zip(["W","J"], axesp))
            qq = stats_results["postcue→path_length2"][subject]
            rsstot = qq["rss"]
            r²pcpl = qq["r²"]
            r²pc = stats_results["postcue"][subject]["r²"] 
            r²pl = stats_results["path_length2"][subject]["r²"]
            r²1res = stats_results["postcue→path_length2"][subject]["r²_res"]
            r²2res = stats_results["path_length2→postcue"][subject]["r²_res"]
            qq = stats_results["postcue→avg_speed"][subject]
            r²pcas = qq["r²"]
            r²as = stats_results["avg_speed"][subject]["r²"] 

            barplot!(ax, 1:5, [r²pc, r²pl, r²pcpl, r²as, r²pcas], color=ax.palette.color[][ii])
            hlines!(ax, r²pc+r²pl, color="black", linestyle=:dot)
            hlines!(ax, r²as+r²pc, color="black", linestyle=:dot)

            # speed

            ax.xticks=([1,2,3,4,5], ["MP", "PL", "MP+PL","AS", "MP+AS"])
            ax.xticklabelrotation = π/3
        end
        axesp[1].ylabel = "r²"
        axesp[2].yticklabelsvisible = false
        axesp[2].yticksvisible = false
        axesp[2].leftspinevisible = false

        # add trajectory illustration
        lg4 = GridLayout()
        fig[1,1] = lg4
        plot_trajectory_illustration!(lg4)
        labels = [Label(fig[1,1,TopLeft()], "A"),
                Label(lg2[1,1,TopLeft()], "B"),
                Label(lg2[1,2,TopLeft()], "C"),
                Label(lg2[1,3,TopLeft()], "D"),
                Label(fig[1,2,TopLeft()], "E")]
        ax =axes[1]
        ax.ylabel = "log(rt)"
        ax.xlabel = "Motor prep (MP)"
        ax = axes[2]
        ax.yticklabelsvisible = false
        ax.xlabel = "Path length (PL)"
        ax = axes[3]
        ax.yticklabelsvisible = false
        ax.xlabel = "Avg speed (AS)"
        rowsize!(fig.layout,1,Relative(0.25))
        fname = joinpath("figures","manuscript","reaction_time_regressors_new.pdf")
        if get(Dict(kvs), :do_save, false)
            save(fname, fig,pt_per_unit=1)
        end
        rowgap!(fig.layout, 1, 5)
        colsize!(fig.layout, 1, Relative(0.6))
        fig
    end
end

minimum_nan(x) = minimum(filter(isfinite, x))

function plot_regression(stats_results, v1::String, v2::String, subject::String)
    fig = Figure()
    ax = Axis3(fig[1,1])
    β = stats_results["$(v1)→$(v2)"][subject]["β"]
    β1 = stats_results[v1][subject]["β"]
    S1 = repeat(stats_results[v1][subject]["S"],1,1)

    β2 = stats_results[v2][subject]["β"]
    S2 = repeat(stats_results[v2][subject]["S"],1,1)
    S = [S1 S2]
    rtp = S*β[1:2] .+ β[3]
    lrt = stats_results[v1][subject]["lrt"]
    scatter!(ax, S1[:], S2[:], rtp, label="$(v1)+$(v2)")
    scatter!(ax, S1[:], fill(minimum_nan(S2), length(S1)), S1*β1[1:1] .+ β1[2], label=v1)
    scatter!(ax, fill(minimum_nan(S1), length(S2)), S2[:], S2*β2[1:1] .+ β2[2], label=v2)

    scatter!(ax, S1[:], S2[:], lrt, label="Actual")

    # single regression
    _color = ax.palette.color[][4]
    scatter!(ax, S1[:], fill(minimum_nan(S2), length(S1)), lrt,color=(_color, 0.5))
    scatter!(ax, fill(minimum_nan(S1), length(S2)), S2[:],  lrt, color=(_color, 0.5))
    ax.xlabel = v1
    ax.ylabel = v2 
    ax.zlabel = "log(rt)"
    Legend(fig[1,2], ax)
    fig,ax
end

function plot_regression_error(stats_results, v1, v2, subject)
    β = stats_results["$(v1)→$(v2)"][subject]["β"]
    β1 = stats_results[v1][subject]["β"]
    S1 = repeat(stats_results[v1][subject]["S"],1,1)

    β2 = stats_results[v2][subject]["β"]
    S2 = repeat(stats_results[v2][subject]["S"],1,1)
    S = [S1 S2]
    rtp = S*β[1:2] .+ β[3]
    lrt = stats_results[v1][subject]["lrt"]
    rtp1 = S1*β1[1:1] .+ β1[2]
    rtp2 = S2*β2[1:1] .+ β2[2]

    ymin = min(minimum(lrt), minimum(rtp2))
    ymax = max(maximum(lrt), maximum(rtp2))
    points = [Point2(_lrt, _prt) for (_lrt,_prt) in zip(lrt,rtp2)]
    directions = [Point2(1.0, -1.0)*(_prt-_lrt)/sqrt(2) for (_lrt, _prt) in zip(lrt, rtp2)]
    with_theme(plot_theme) do
        fig = Figure()
        ax = Axis(fig[1,1], aspect=1.0)
        #scatter!(ax, lrt, rtp, label="$(v1)+$(v2)")
        #scatter!(ax, lrt, rtp1 , label=v1)
        arrows!(ax, points, directions)
        scatter!(ax, lrt, rtp2 , label=v2)
        ablines!(ax, 0.0, 1.0, linestyle=:dot, color="black")
        xlims!(ax, ymin,ymax)
        ylims!(ax, ymin,ymax)
        axislegend(ax)
        fig
    end
end

end # module