module CSIAnalysis
using StatsBase
using DemixedPCA
using JLD2
using Distributions
using DataProcessingHierarchyTools
using Makie
using Colors
using LinearRegressionUtils
const DPHT =DataProcessingHierarchyTools 

using ..Utils
using ..PlotUtils

function run_analysis(sessionidx;window::Union{Nothing, Float64}=nothing)
    ppstht, labelst, rtimest,trialidxt = JLD2.load(joinpath("..","PrefrontalManifoldGeometry","data","ppsth_fef_cue.jld2"), "ppsth", "labels","rtimes","trialidx");
    all_sessions = DPHT.get_level_path.("session", ppstht.cellnames)
    sessions = unique(all_sessions)
    session = sessions[sessionidx]

    X, _labels, _rtimes = Utils.get_session_data(session, ppstht, trialidxt, labelst, rtimest;mean_subtract=false, variance_stabilize=false,rtime_min=120.0, rtime_max=300.0)
    nbins,ntrials,ncells = size(X)
    if window !== nothing
        #X2,bins = Utils.rebin2(X, ppstht.bins, window)
        X2 = mapslices(x->Utils.gaussian_smooth(x,ppstht.bins, window), X, dims=1)        
    else
        X2 = X
        bins = ppstht.bins
    end
    Xp = permutedims(X2,[3,2,1])
    # now run demixed PCA
    dmpca = fit(dPCA, Xp, _labels) 
    Y = DemixedPCA.transform(dmpca["time"],Xp)
    dmpca, Y,ppstht.bins, _rtimes
end

function plot(Y::Array{T,3},bins) where T <: Real
    # compute average over trials
    μ = dropdims(mean(Y,dims=2),dims=2)
    with_theme(PlotUtils.plot_theme) do 
        fig = Figure()
        axes = [Axis(fig[i,1]) for i in 1:size(μ,1)]
        for (i,ax) in enumerate(axes)
            vlines!(ax, 0.0, color=:black, linestyle=:dot)
            lines!(ax, bins, μ[i,1:length(bins)])
        end
        for ax in axes[1:end-1]
            ax.xticklabelsvisible=false
            ax.bottomspinevisible = false
            ax.xticksvisible = false
        end
        ax = axes[end]
        ax.xlabel = "Time from go-cue"
        linkxaxes!(axes...)
        fig
    end
end

function find_crossings(Y::Matrix{T}, bins, rt;rt_percentile_bin=10.0) where T <: Real
    idx0 = searchsortedfirst(bins, 0.0)
    rtbins = range(0.0, step=rt_percentile_bin, stop=100.0)
    colors = resample_cmap(:thermal, length(rtbins))
    sidx = sortperm(rt)
    vidx = invperm(sidx)
    qidx = 99*(vidx .- minimum(vidx))./(maximum(vidx)-minimum(vidx)) .+ 1 
    rtlabel = fill(0, length(rt))
    rtq = fill(0.0, length(rtbins))
    crossing = fill(NaN, length(rtlabel))
    for i in eachindex(rtlabel)
        rtlabel[i] = searchsortedlast(rtbins, qidx[i])
    end
    θ = fill(0.0, length(rtq))
    for i in 1:length(rtq)
        tidx = rtlabel .== i
        rtq[i] = mean(rt[tidx])
        μ = dropdims(mean(Y[tidx,1:length(bins)],dims=1),dims=1)
        μ0 = mean(μ[1:idx0-1])
        idxe = searchsortedfirst(bins, rtq[i])
        im = argmax(abs.(μ[idx0:idxe] .- μ0))
        θ[i] = μ0 + 0.5*(μ[idx0+im-1]-μ0)
    end

    colors2 = colors[rtlabel]
    for i in eachindex(rtlabel)
        j = rtlabel[i]
        if θ[j] < 0.0
            _bidx = findfirst(Y[i,idx0:end].<θ[j])
        else
            _bidx = findfirst(Y[i,idx0:end].>θ[j])
        end

        if _bidx === nothing
            continue
        end
        crossing[i] = bins[idx0+_bidx-1]
        if crossing[i] > rt[i]
            cc = colors2[i]
            colors2[i] = RGBA(cc.r, cc.g, cc.b, 0.25)
        end
    end
    rridx = isfinite.(crossing)
    lreg = LinearRegressionUtils.llsq_stats(repeat(crossing[rridx],1,1),rt[rridx])
    crossing[rridx], rt[rridx], lreg
end

function plot!(lg;kvs...)
    dmpca,Y,bins,rt = run_analysis(5,;window=20.0)
    plot!(lg, Y[1,:,:], bins, rt;kvs...)
end

function plot(;kvs...)
    with_theme(plot_theme) do
        fig = Figure()
        lg = GridLayout(fig[1,1])
        plot!(lg;kvs...)
        fig
    end
end

function plot!(lg, Y::Matrix{T}, bins, rt::AbstractVector{T};rt_percentile_bin=10.0,per_percentile_threshold=true,tmin=bins[1],tmax=bins[end],show_equality_line=false, ylabel="CSI 1") where T <: Real
    # set a threshold equal to 50% of the global maximum
    bidx = searchsortedfirst(bins, tmin):searchsortedlast(bins, tmax)
    idx0 = searchsortedfirst(bins, 0.0)
    rtbins = range(0.0, step=rt_percentile_bin, stop=100.0)
    colors = resample_cmap(:thermal, length(rtbins))
    sidx = sortperm(rt)
    vidx = invperm(sidx)
    qidx = 99*(vidx .- minimum(vidx))./(maximum(vidx)-minimum(vidx)) .+ 1 
    rtlabel = fill(0, length(rt))
    rtq = fill(0.0, length(rtbins))
    crossing = fill(NaN, length(rtlabel))
    for i in eachindex(rtlabel)
        rtlabel[i] = searchsortedlast(rtbins, qidx[i])
    end
    θ = fill(0.0, length(rtq))
    ymin = Inf
    μ = fill(0.0, length(bins), length(rtq))
    for i in 1:length(rtq)
        tidx = rtlabel .== i
        rtq[i] = mean(rt[tidx])
        μ[:,i] = dropdims(mean(Y[tidx,1:length(bins)],dims=1),dims=1)
        ymin = min(ymin, minimum(μ[:,i]))
    end
    if per_percentile_threshold
        for i in 1:length(rtq)
            μ0 = mean(μ[1:idx0-1,i])
            idxe = searchsortedfirst(bins, rtq[i])
            im = argmax(abs.(μ[idx0:idxe,i] .- μ0))
            θ[i] = μ0 + 0.5*(μ[idx0+im-1,i]-μ0)
        end
    else
        _μ = dropdims(mean(μ,dims=2),dims=2)
        μ0 = mean(_μ[1:idx0-1])
        im = argmax(abs.(_μ[idx0:end] .- μ0))
        θ .= μ0 + 0.5*(_μ[idx0+im-1]-μ0)
    end

    colors2 = colors[rtlabel]
    for i in eachindex(rtlabel)
        j = rtlabel[i]
        if θ[j] < 0.0
            _bidx = findfirst(Y[i,idx0:end].<θ[j])
        else
            _bidx = findfirst(Y[i,idx0:end].>θ[j])
        end

        if _bidx === nothing
            continue
        end
        crossing[i] = bins[idx0+_bidx-1]
        if crossing[i] > rt[i]
            crossing[i] = NaN
            cc = colors2[i]
            colors2[i] = RGBA(cc.r, cc.g, cc.b, 0.25)
        end
    end
    rridx = isfinite.(crossing)
    lreg = LinearRegressionUtils.llsq_stats(repeat(crossing[rridx],1,1),rt[rridx])
    @show lreg.r², lreg.pv
    with_theme(PlotUtils.plot_theme)  do
        ax = Axis(lg[1,1:2])
        vlines!(ax, 0.0, color=:black, linestyle=:dot)
        for i in 1:length(rtbins)
            tidx = rtlabel .== i
            if per_percentile_threshold
                hlines!(ax, θ[i], color=colors[i], linestyle=:dot)
            end
            μ = dropdims(mean(Y[tidx,1:length(bins)],dims=1),dims=1)
            lines!(ax, bins[bidx], μ[bidx],color=colors[i])
            hidx = searchsortedfirst(bins, rtq[i])
            if 0 < hidx <= length(bins)
                linesegments!(ax, [Point2f(bins[hidx],ymin)=>Point2f(bins[hidx], μ[hidx])],
                                color=colors[i])
            end
        end
        if !per_percentile_threshold
            hlines!(ax, θ[1], color=:black, linestyle=:dot)
        end
        cb = Colorbar(fig[1,2],colormap=:thermal, colorrange=(extrema(rt)...,),label="Reaction time")
        ax.xlabel = "Time from go-cue onset"
        ax2 = Axis(fig[2,1:2])
        scatter!(ax2, crossing, rt,color=colors2)
        if show_equality_line
            linesegments!(ax2, [Point2(minimum(rt))=>Point2(maximum(rt))],color=:black)
        end
        ablines!(ax2, lreg.β[2], lreg.β[1],color=:red)
        ax2.ylabel = "Reaction time"
        ax2.xlabel = "Crossing"
        fig
    end
end

function plot2(Y::Matrix{T}, bins, rt::AbstractVector{T};rt_percentile_bin=10.0) where T <: Real
    # set a threshold equal to 50% of the global maximum
    μ = dropdims(mean(Y,dims=1),dims=1)
    μm,im  = findmax(abs.(μ))
    idx0 = searchsortedfirst(bins, 0.0)
    μ0 = mean(μ[1:idx0-1])
    θ = μ0 +  0.5*(μ[im]-μ0)
    rtbins = range(0.0, step=rt_percentile_bin, stop=100.0)
    colors = resample_cmap(:thermal, length(rt))
    sidx = sortperm(rt)
    vidx = invperm(sidx)
    qidx = 99*(sidx .- minimum(sidx))./(maximum(sidx)-minimum(sidx)) .+ 1 
    rtlabel = fill(0, length(rt))
    rtq = fill(0.0, length(rtbins))
    crossing = fill(NaN, length(rtlabel))
    ccidx = fill(0, length(rt))
    for i in eachindex(rtlabel)
        rtlabel[i] = searchsortedlast(rtbins, qidx[i])
        if θ < 0.0
            _bidx = findfirst((Y[i,idx0:end]).<θ)
        else
            _bidx = findfirst((Y[i,idx0:end]).>θ)
        end
        if _bidx === nothing
            continue
        end
        ccidx[i] = idx0+_bidx-1
        crossing[i] = bins[ccidx[i]]
    end
    ymin = minimum(Y)
    with_theme(PlotUtils.plot_theme)  do
        fig = Figure()
        ax = Axis3(fig[1,1])
        #vlines!(ax, 0.0, color=:black, linestyle=:dot)
        for i in 1:length(rtlabel)
            #hlines!(ax, θ, color=:blue, linestyle=:dot)
            lines!(ax, bins, fill(i, length(bins)), Y[sidx[i],1:length(bins)],color=colors[i])
            if isfinite(crossing[sidx[i]])
                scatter!(ax, crossing[sidx[i:i]], [i], [ymin], color=colors[i])
                linesegments!(ax, [Point3(crossing[sidx[i]], i, ymin)=>Point3(crossing[sidx[i]],i, Y[sidx[i],ccidx[sidx[i]]])],color=colors[i])
            end
        end
        cb = Colorbar(fig[1,2],colormap=:thermal, colorrange=(extrema(rt)...,),label="Reaction time")
        ax.xlabel = "Time from go-cue onset"
        fig
    end
end
end #module