using StatsBase
using DemixedPCA
using JLD2
using Distributions
using DataProcessingHierarchyTools
using Makie
using Colors
using LinearRegressionUtils
const DPHT =DataProcessingHierarchyTools 

include("utils.jl")
include("plot_utils.jl")

function gaussian_smooth(X::AbstractVector{T}, bins::AbstractVector{T},σ::T) where T <: Real
    Δ = mean(diff(bins))
    Y = fill(0.0, length(X))
    for i in 1:length(bins)
        l = bins[end]-bins[i]
        u = bins[1]-bins[i]
        N = TruncatedNormal(0.0, σ,min(l,u) ,max(l,u))
        Y[i] = sum(pdf.(N, bins[i] .- bins).*X*Δ)
    end
    Y
end

function run(sessionidx;window::Union{Nothing, Float64}=nothing)
    ppstht, labelst, rtimest,trialidxt = JLD2.load(joinpath("..","PrefrontalManifoldGeometry","data","ppsth_fef_cue.jld2"), "ppsth", "labels","rtimes","trialidx");
    all_sessions = DPHT.get_level_path.("session", ppstht.cellnames)
    sessions = unique(all_sessions)
    session = sessions[sessionidx]

    X, _labels, _rtimes = Utils.get_session_data(session, ppstht, trialidxt, labelst, rtimest;mean_subtract=false, variance_stabilize=false,rtime_min=120.0, rtime_max=300.0)
    if window !== nothing
        #X2,bins = Utils.rebin2(X, ppstht.bins, window)
        X2 = mapslices(x->gaussian_smooth(x,ppstht.bins, window), X, dims=1)        
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

function plot(Y::Matrix{T}, bins, rt::AbstractVector{T};rt_percentile_bin=10.0) where T <: Real
    # set a threshold equal to 50% of the global maximum
    μ = dropdims(mean(Y,dims=1),dims=1)
    μm,im  = findmax(abs.(μ))
    idx0 = searchsortedfirst(bins, 0.0)
    μ0 = mean(μ[1:idx0-1])
    θ = μ0 +  0.5*(μ[im]-μ0)
    rtbins = range(0.0, step=rt_percentile_bin, stop=100.0)
    colors = resample_cmap(:thermal, length(rtbins))
    sidx = sortperm(rt)
    vidx = invperm(sidx)
    qidx = 99*(vidx .- minimum(vidx))./(maximum(vidx)-minimum(vidx)) .+ 1 
    rtlabel = fill(0, length(rt))
    crossing = fill(NaN, length(rtlabel))
    for i in eachindex(rtlabel)
        rtlabel[i] = searchsortedlast(rtbins, qidx[i])
    end
    colors2 = colors[rtlabel]
    for i in eachindex(rtlabel)
        if θ < 0.0
            _bidx = findfirst(Y[i,idx0:end].<θ)
        else
            _bidx = findfirst(Y[i,idx0:end].>θ)
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
    @show lreg.r², lreg.pv
    with_theme(PlotUtils.plot_theme)  do
        fig = Figure()
        ax = Axis(fig[1,1])
        vlines!(ax, 0.0, color=:black, linestyle=:dot)
        for i in 1:length(rtbins)
            tidx = rtlabel .== i
            hlines!(ax, θ, color=:blue, linestyle=:dot)
            lines!(ax, bins, dropdims(mean(Y[tidx,1:length(bins)],dims=1),dims=1),color=colors[i])
        end
        cb = Colorbar(fig[1,2],colormap=:thermal, colorrange=(extrema(rt)...,),label="Reaction time")
        ax.xlabel = "Time from go-cue onset"
        ax2 = Axis(fig[2,1:2])
        scatter!(ax2, crossing, rt,color=colors2)
        linesegments!(ax2, [Point2(minimum(rt))=>Point2(maximum(rt))],color=:black)
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