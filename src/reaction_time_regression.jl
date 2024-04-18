using MultivariateStats
using JLD2
using LinearRegressionUtils
using Distributions
using CairoMakie

include("utils.jl")
include("trajectories.jl")
include("plot_utils.jl")

function balance_num_trials(label::Vector{Int64}, args...)
    cc = countmap(label)
    nn,_ = findmin(cc)
    idxt = fill(0, nn*length(cc))
    for (i,l) in enumerate(sort(collect(keys(cc))))
        idx = findall(label.==l)
        idxt[(i-1)*nn+1:i*nn] .= sort(shuffle(idx)[1:nn])
    end
    oargs = Any[label[idxt]]
    for _arg in args
        if ndims(_arg) == 2
            push!(oargs, _arg[idxt,:])
        else
            push!(oargs, _arg[idxt])
        end
    end
    push!(oargs, idxt)
    oargs
end

"""
    get_regression_data(subject;area="fef", rtmin=120.0, rtmax=300.0, window=35.0, Δt=15.0,align=:mov, realign=true, raw=false, do_shuffle=false, nruns=100,smooth_window::Union{Nothing, Float64}=nothing, kvs...)

Get data for regressing reaction time for each point in time for the specified `subject` and `area`.
"""
function get_regression_data(subject;area="fef", rtmin=120.0, rtmax=300.0, window=35.0, Δt=15.0,align=:mov, realign=true, raw=false, do_shuffle=false, nruns=100,smooth_window::Union{Nothing, Float64}=nothing, kvs...)
    if raw
        fname = joinpath("data","ppsth_$(area)_$(align)_raw.jld2")
    else 
        fname = joinpath("data","ppsth_$(area)_$(align).jld2")
    end
    ppsth,tlabels,trialidx, rtimes = JLD2.load(fname, "ppsth","labels","trialidx","rtimes")

	# Per session, per target regression, combine across to compute rv
	all_sessions = Utils.DPHT.get_level_path.("session", ppsth.cellnames)
    subjects = Utils.DPHT.get_level_name.("subject", ppsth.cellnames)
    cidx = subjects .== subject
	sessions = unique(all_sessions[cidx])
	rtp = fill(0.0, 0, size(ppsth.counts,1))
	rt = fill(0.0, 0)
    label = fill(0,0)
    ncells = fill(0,0)
	bins = ppsth.bins
	rtime_all = Float64[]
    r2j = fill(0.0, size(ppsth.counts,1),100)
    total_idx = 1
    n_tot_trials =  sum([length(rtimes[k]) for k in sessions])
    Z = fill(0.0,n_tot_trials, size(ppsth.counts,1))
    L = fill!(similar(Z),NaN) 
    offset = 0
	for (ii, session) in enumerate(sessions)
		X, _label, _rtime = Utils.get_session_data(session,ppsth, trialidx, tlabels, rtimes;rtime_min=rtmin,rtime_max=rtmax,kvs...)
        _ncells = size(X,3)
        _lrt = log.(_rtime)
        _nt = length(_rtime)
        #if smooth_window !== nothing
        #    X2 = mapslices(x->Utils.gaussian_smooth(x,bins,smooth_window), X, dims=1)
        #else
        #    X2 = X
        #end
        for j in 1:size(X,1)
            idx0 = j
            idx1 = searchsortedlast(bins, bins[j]+window)
            # project onto FA components
            y = permutedims(dropdims(sum(X[idx0:idx1,:,:],dims=1),dims=1))
            fa = StatsBase.fit(MultivariateStats.FactorAnalysis,y;maxoutdim=1, method=:em)
            z =  permutedims(predict(fa, y))
            if realign
                # check the sign of the correlation between reaction time and activity and realign so that the relationship is positive
                β = llsq(z, _lrt)
                if β[1] < 0.0
                    z .*= -1.0
                end
            end
            Z[offset+1:offset+_nt,j]  .= z
            # path length
            for j2 in 1:_nt
                idxq = searchsortedlast(bins, bins[idx1]+Δt)
                idxs = searchsortedlast(bins, bins[idx0]+_rtime[j2])
                if idxs >= length(bins)
                    continue
                end
                # transition period
                # midpoint
                ip = div(idxs-idxq+1,2)+1

                Xs = X[idxq:idxs,j2,:]
                # point of high energy
                ee = dropdims(sum(abs2, Xs,dims=2),dims=2)
                ip = argmax(ee)
                if smooth_window !== nothing
                    Xs = mapslices(x->Utils.gaussian_smooth(x, bins[idxq:idxs], smooth_window), Xs, dims=1)
                end
                L[offset+j2,j] = compute_triangular_path_length(Xs, ip)
                #L[offset+j2,j] /=_ncells
            end
        end
        append!(rt, _lrt)
        append!(label, _label)
        append!(ncells, fill(size(X,3), length(_label)))
        offset += _nt
    end
    Z[1:offset,:], L[1:offset,:], rt, label, ncells, bins
end

"""

Compute
"""
function compute_regression(Z::Matrix{Float64}, L::Matrix{Float64}, xpos::Vector{Float64}, ypos::Vector{Float64}, ncells::Vector{Int64}, rt::Vector{Float64},bins::AbstractVector{Float64})
    nbins = size(Z,2)
    β = fill(NaN, 5, nbins)
    Δβ = fill(NaN, 5, nbins)
    pv = fill(NaN, nbins)
    r² = fill(NaN, nbins)
    for i in axes(Z,2) 
        if !all(isfinite.(L[:,i]))
          continue
        end
        # run two regression models; one without path length L and one with
        X_no_L = [Z[:,i] xpos ypos ncells]
        X_with_L = [Z[:,i] xpos ypos L[:,i] ncells] 

        lreg_no_L = LinearRegressionUtils.llsq_stats(X_no_L, rt;do_interactions=true, exclude_pairs=[(2,3)])
        lreg_with_L = LinearRegressionUtils.llsq_stats(X_with_L, rt;do_interactions=true, exclude_pairs=[(2,3)])

        # compute the F-stat for whether adding the path length results in a significantly better fit
        n = length(rt)
        rss2 = lreg_with_L.rss
        p2 = length(lreg_with_L.β)
        rss1 = lreg_no_L.rss
        p1 = length(lreg_no_L.β)
        if i == 1
            @show p1 p2 rss1 rss2 lreg_with_L.varidx
        end
        F = (rss1 - rss2)/(p2-p1)
        F /= rss2/(n-p2)
        pv[i] = 1.0 - cdf(FDist(p2-p1, n-p2), F)
        r²[i] = lreg_with_L.r²
        β[1,i] = lreg_with_L.β[4]
        Δβ[1,i] = lreg_with_L.Δβ[4]
        vidx = findall(ii->(length(ii)==2)&&((ii[1]==4)||(ii[2]==4)), lreg_with_L.varidx)
        β[2:end,i] = lreg_with_L.β[vidx]
        Δβ[2:end,i] = lreg_with_L.Δβ[vidx]
    end
    β,Δβ, pv, r²
end

function plot_regression(β, Δβ,pv,r²,bins)
    pidx = findall(pv .< 0.05)
    iidx = findall(isfinite, pv)
    with_theme(PlotUtils.plot_theme) do
        fig  = Figure()
        ax1 = Axis(fig[1,1])
        lines!(ax1, bins[iidx],r²[iidx])
        scatter!(ax1, bins[pidx], fill(1.01*maximum(filter(isfinite,r²)), length(pidx)), color=:red)
        axes = [Axis(fig[1+i,1]) for i in 1:size(β,1)]
        linkxaxes!(axes...)
        for (i,ax2) in enumerate(axes)
            lines!(ax2,bins[iidx],β[i,iidx])
            band!(ax2, bins[iidx],(β[i,:]-2Δβ[i,:])[iidx], (β[i,:]+2Δβ[i,:])[iidx])
            hlines!(ax2, 0.0, color=:black, linestyle=:dot)
        end

        for ax in [ax1,axes...]
            vlines!(ax, 0.0, color=:black)
        end
        for ax in [ax1, axes[1:end-1]...]
            ax.xticklabelsvisible = false
        end
        fig
    end
end

function plot_fef_dlpfc_r²()
    fname = "path_length_regression_fef_dlpfc_comparison.jld2"
    if isfile(fname)
        qdata = JLD2.load(fname)
    else
        qdata = Dict()
        for subject in ["J","W"]
            Z,L,lrt,label,ncells,bins = get_regression_data(subject;area="fef", align=:cue,raw=true, mean_subtract=true, variance_stabilize=true,window=50.0,smooth_window=nothing);
            if subject == "J"
                tidx = findall(label.!=9)
            else
                tidx = 1:size(Z,1)
            end
            xpos = [p[1] for p in Utils.location_position[subject]][label[tidx]]
            ypos = [p[2] for p in Utils.location_position[subject][label[tidx]]]
            βfef,Δβfef,pvfef,r²fef = compute_regression(Z[tidx,:], L[tidx,:], xpos, ypos,ncells[tidx],lrt[tidx],bins)

            Z,L,lrt,label,ncells,bins = get_regression_data(subject;area="dlpfc", align=:cue,raw=true, mean_subtract=true, variance_stabilize=true,window=50.0,smooth_window=nothing);
            βdlpfc,Δβdlpfc,pvdlpfc,r²dlpfc = compute_regression(Z[tidx,:], L[tidx,:], xpos, ypos,ncells[tidx],lrt[tidx],bins)
            qdata[subject] = Dict("βfef"=>βfef, "Δβfef"=>Δβfef,"pvfef"=>pvfef, "r²fef"=>r²fef,
                                  "βdlpfc"=>βdlpfc, "Δβdlpfc"=>Δβdlpfc, "pvdlpfc"=>pvdlpfc, "r²dlpfc"=>r²dlpfc,
                                  "bins"=>bins)
        end
        JLD2.save(fname, qdata)
    end
    with_theme(PlotUtils.plot_theme) do
        fig = Figure(size=(500,500))
        axes = [Axis(fig[i,1]) for i in 1:4]
        for (ii,subject) in enumerate(["J","W"])
            bins,r²fef, r²dlpfc,βfef, Δβfef,βdlpfc, Δβdlpfc = [qdata[subject][k] for k in ["bins", "r²fef","r²dlpfc","βfef","Δβfef","βdlpfc","Δβdlpfc"]]
            bidx = findall(isfinite, r²fef)
            ax1 = axes[(ii-1)*2+1]
            vlines!(ax1, 0.0, color=:black)
            lines!(ax1, bins, r²fef, label="FEF")
            lines!(ax1, bins, r²dlpfc, label="DLPFC")
            ax1.title="Monkey $subject"
            ax1.ylabel = "r²"
            ax2 = axes[ii*2]
            ax2.ylabel = "β"
            lines!(ax2, bins[bidx], βfef[1,bidx], label="FEF")
            band!(ax2, bins[bidx], (βfef-2Δβfef)[1,bidx], (βfef+2Δβfef)[1,bidx])
            lines!(ax2, bins[bidx], βdlpfc[1,bidx], label="DLPFC")
            band!(ax2, bins[bidx], (βdlpfc-2Δβdlpfc)[1,bidx], (βdlpfc+2Δβdlpfc)[1,bidx])
            hlines!(ax2, 0.0, color=:black, linestyle=:dot)

        end
        linkaxes!(axes[1],axes[3])
        linkxaxes!(axes...)
        for ax in axes[1:end-1]
            ax.xticklabelsvisible = false
        end
        ax = axes[end-1]
        ax.xlabel = "Time from go-cue"
        axislegend(ax, halign=:left)
        fig
    end
end

function plot_regression(Z::Matrix{Float64}, L::Matrix{Float64}, xpos::Vector{Float64}, ypos::Vector{Float64}, ncells::Vector{Int64}, rt::Vector{Float64},bins;include_L=true, include_ncells=true, go_cue_onset=2600.0)
    nbins = size(Z,2)
    r² = fill(NaN,nbins)
    pv = fill(NaN,nbins)
    βz = fill(NaN,nbins)
    βl = fill(NaN,nbins)
    βx = fill(NaN,nbins)
    βy = fill(NaN,nbins)
    βnc = fill(NaN, nbins)

    Δβz = fill(NaN,nbins)
    Δβl = fill(NaN,nbins)
    Δβx = fill(NaN,nbins)
    Δβy = fill(NaN,nbins)
    Δβnc = fill(NaN,nbins)
    for i in 1:nbins
        if !all(isfinite.(L[:,i]))
          continue
        end
        if include_L
            X = [Z[:,i] xpos ypos L[:,i]]
        else
            X = [Z[:,i] xpos ypos]
        end
        if include_ncells
            X = [X ncells]
        end
        X_no_L = [Z[:,i] xpos ypos ncells]
        X_with_L = [Z[:,i] xpos ypos L[:,i] ncells] 
        lreg = LinearRegressionUtils.llsq_stats(X, rt;do_interactions=true, exclude_pairs=[(2,3)])
        r²[i] = lreg.r²
        pv[i] = lreg.pv
        βz[i] = lreg.β[1]
        Δβz[i] = lreg.Δβ[1]
        βx[i] = lreg.β[2]
        Δβx[i] = lreg.Δβ[2]
        βy[i] = lreg.β[3]
        Δβy[i] = lreg.Δβ[3]
        if include_L
            βl[i] = lreg.β[4]
            Δβl[i] = lreg.Δβ[4]
        end
        if include_ncells
            βnc[i] = lreg.β[5]
            Δβnc[i] = lreg.Δβ[5]
        end
    end
    pidx = findall(pv .< 0.01)
    bidx = findall(isfinite, r²)
    with_theme(PlotUtils.plot_theme) do
        fig = Figure(size=(500,600))
        if include_L
            βs =  [βz, βl, βx, βy] 
            Δs = [Δβz, Δβl, Δβx, Δβy] 
            ylabel = ["FA","PL","X","Y"]
            axes = [Axis(fig[i,1]) for i in 1:5]
        else
            βs =  [βz, βx, βy] 
            Δs = [Δβz, Δβx, Δβy] 
            ylabel = ["FA","X","Y"]
            axes = [Axis(fig[i,1]) for i in 1:4]
        end
        if include_ncells
            push!(βs, βnc)
            push!(Δs, Δβnc)
            push!(ylabel, "ncells")
            push!(axes, Axis(fig[length(axes)+1,1]))
        end
        linkxaxes!(axes...)

        ax = axes[1]
        ax.xticksvisible = true
        ax.xticklabelsvisible = false
        lines!(ax, bins, r²)
        scatter!(ax, bins[pidx], fill(maximum(filter(isfinite, r²)), length(pidx)),color=:red)
        if go_cue_onset < bins[end]
            vlines!(ax, go_cue_onset, color=:grey, linestyle=:dot)
        end
        ax.ylabel = "r²"
        for (ax,β,Δ,l) in zip(axes[2:end],βs,Δs,ylabel)
            lines!(ax, bins[bidx], β[bidx])
            band!(ax, bins[bidx], (β-2Δ)[bidx], (β+2Δ)[bidx])
            hlines!(ax, 0.0, linestyle=:dot, color=:black)
            if go_cue_onset < bins[end]
                vlines!(ax, go_cue_onset, color=:grey, linestyle=:dot)
            end
            ax.ylabel = "β $l"
            ax.xticksvisible = true
            ax.xticklabelsvisible = false
        end
        ax = axes[end]
        ax.xticksvisible = true
        ax.xticklabelsvisible = true
        ax.xlabel = "Time"
        fig
    end
end

function plot_β_summary()
    fname = "path_length_regression_fef_dlpfc_comparison.jld2"
    if isfile(fname)
        qdata = JLD2.load(fname)
    end
    μ = Dict()
    for (ii,subject) in enumerate(["J","W"])
        bins, βfef, Δβfef,βdlpfc, Δβdlpfc = [qdata[subject][k] for k in ["bins","βfef","Δβfef","βdlpfc","Δβdlpfc"]]
        idx0 = searchsortedfirst(bins,-300.0)        
        idx1 = searchsortedfirst(bins, -150.0)
        idx2 = searchsortedfirst(bins, 65);
        μdlpfc = abs(βdlpfc[1,idx2])/abs(mean(βdlpfc[1,idx0:idx1]))
        μfef = abs(βfef[1,idx2])/abs(mean(βfef[1,idx0:idx1]))
        μ[subject] = Dict("dlpfc"=>μdlpfc,"fef"=>μfef)
    end

    colors = [PlotUtils.fef_color, PlotUtils.dlpfc_color]
    with_theme(PlotUtils.plot_theme) do
        fig = Figure()
        ax = Axis(fig[1,1])

        barplot!(ax, 1:4, [μ["J"]["fef"], μ["J"]["dlpfc"],μ["W"]["fef"],μ["W"]["dlpfc"]],
                          color=colors[[1,2,1,2]])
        Legend(fig[1,1], [PolyElement(color=colors[i]) for i in 1:2],["FEF", "DLPFC"],
                valign=:top, halign=:left, tellwidth=false)
        ax.ylabel = "β transition/β baseline"
        ax.xticks=([1.5, 3.5],["Monkey J","Monkey W"])
        fig
    end
end

"""
A simple model trying to distinguish integrating noise over different lengths of time
vs actual differences in path length. 

If the results arise because we are integrating noise, then shuffling each dimension independently
should not alter the results (given that the noise is the same, I guess.)
However, if there is actual structure, then shuffling should break the correlation.
"""
function simple_model()
    # path lengths
    n1 = 11
    n2 = 19

end