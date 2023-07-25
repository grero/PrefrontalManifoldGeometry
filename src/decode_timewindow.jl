module TimeWindowDeocding
using LinearAlgebra
using StatsBase
using MultivariateStats
using EventOnsetDecoding
using Random
using LinearAlgebra
using Makie
using Makie: Colors
using Colors
using CairoMakie

include("utils.jl")

"""
    decode_window(X::Matrix{T}, bins::AbstractVector{Float64}, window::Tuple{Float64})

Train an LDA decoder to seperate the activity during the specified window from any other window.
"""
function decode_window(X::Array{T,3}, bins::AbstractVector{Float64}, window::Tuple{Float64,Float64}) where T <: Real
    nbins, ntrials, ncells = size(X)
    idx1 = searchsortedfirst(bins, window[1])
    idx2 = searchsortedlast(bins, window[2])
    w = idx2-idx1+1
    Xm = dropdims(sum(X[idx1:idx2, :,:], dims=1),dims=1)

    # we want a decoder that can tell the responses from `Xm` from all the other responses in `X`
    Xn = fill!(similar(Xm), 0.0)
    for i in axes(Xm,1)
        #grab a random bin
        idxt = rand(1:size(X,1)-w)
        idxe = idxt+w-1
        Xn[i,:] .= dropdims(sum(X[idxt:idxe,i,:],dims=1),dims=1)
    end
    Xc = permutedims([Xm;Xn])
    ll = [fill(1,ntrials);fill(2,ntrials)]
    lda = fit(SubspaceLDA, Xc, ll)
end

function decode_window(ppsth, trialidx::Vector{Vector{Int64}}, tlabels::Vector{Vector{Int64}}, rtimes, window;area::String="FEF", locations::Union{Nothing, Vector{Int64}}=nothing, rtime_min=120.0, rtime_max=300.0, combine_locations=true,nruns=100)
    bins = ppsth.bins
    allsessions = DPHT.get_level_path.("session", ppsth.cellnames)
    sessions = unique(allsessions)
    sort!(sessions)
    cellidx = findall(get_area_index(ppsth.cellnames, "FEF"))
    Xtot = fill(0.0, size(ppsth.counts,1), size(ppsth.counts, 2), length(cellidx))
    label_tot = Vector{Vector{Int64}}(undef, length(cellidx))
    celloffset = 0
    for i in 1:length(sessions) 
        X, _label, _rtime = EventOnsetDecoding.get_session_data(sessions[i],ppsth, trialidx, tlabels, rtimes, cellidx;rtime_min=rtime_min,rtime_max=rtime_max)
        if locations !== nothing
            ttidx = findall(in(locations), _label)
            _label = [findfirst(locations.==l) for l in _label[ttidx]]
        else
            ttidx = collect(eachindex(_label)) 
        end
        X = X[:, ttidx,:]
        _rtime = _rtime[ttidx]
        if combine_locations
            fill!(_label, 1)
        end
        _ncells = size(X,3)
        cidx = celloffset+1:celloffset+_ncells
        Xtot[:, 1:size(X,2),cidx] .= X
        for (jj, cc) in enumerate(cidx)
            label_tot[cc] = _label
        end
        celloffset += _ncells
    end
    Xtot = Xtot[:,:,1:celloffset]
    label_tot = label_tot[1:celloffset]
    ncells = size(Xtot,3)
    ulabel = union(map(unique, label_tot)...)
    use_locations = ulabel
    sort!(use_locations)
    nlocations = length(use_locations)
    RNGs = [MersenneTwister(rand(UInt32)) for r in 1:nruns]
    # posterior for class 1
    binsize = bins[2]-bins[1]
    wl = round(Int64,(window[2]-window[1])/binsize)
    posterior = fill(0.0, size(Xtot,1)-wl+1, length(RNGs))
    Threads.@threads for r in 1:length(RNGs)
        Yt, train_label,test_label =  EventOnsetDecoding.sample_trials(permutedims(Xtot,[3,2,1]), label_tot;RNG=RNGs[r])
        ntrain,ntest = length.([train_label, test_label])
        Ytest = Yt[:, ntrain+1:end, :]
        Ytrain = Yt[:, 1:ntrain, :]
        # train LDA
        lda = decode_window(Ytrain, bins, window)
        pmeans = predict(lda, classmeans(lda)) 
        # test 
        for j in axes(Ytest,2)
            for i in axes(posterior,1)
                q = predict(lda, permutedims(dropdims(sum(Ytest[i:i+wl-1,j:j,:],dims=1),dims=1)))
                d = dropdims(sum(abs2, q .- pmeans,dims=1),dims=1)
                dm = argmin(d)
                posterior[i,r] +=  (dm==1)
            end 
        end
        posterior[:,r] ./= ntest
   end
   posterior,bins[1:size(posterior,1)]
end

function plot(bins::AbstractVector{Float64}, posterior::Matrix{Float64},window::Tuple{Float64, Float64};kvs...)
    fig = Figure()
    ax = Axis(fig[1,1])
    plot!(ax, bins, posterior,window)
    ax.xlabel = "Time [ms]"
    ax.ylabel = "F₁ score"
    fig
end

function plot!(ax, bins::AbstractVector{Float64}, posterior::Matrix{Float64},window::Tuple{Float64, Float64};kvs...)
    vspan!(ax, window[1], window[2],color=RGB(0.8, 0.8, 0.8))
    μ = dropdims(mean(posterior, dims=2), dims=2)
    lines!(ax, bins, μ)
    ax
end

function plot(bins, posteriors::Vector{Matrix{Float64}}, windows::Vector{Tuple{Float64, Float64}})
    with_theme(theme_minimal()) do
        fig = Figure(resolution=(400,500))
        n = length(posteriors)
        axes = [Axis(fig[i,1], xticksvisible=true, yticksvisible=true) for i in 1:n]
        linkaxes!(axes...)
        for (ax, posterior, window) in zip(axes, posteriors, windows)
            plot!(ax, bins, posterior, window)
        end
        for ax in axes[1:end-1]
            ax.xticklabelsvisible = false
        end
        ax = axes[end]
        ax.xlabel = "Time from movement [ms]"
        ax.ylabel = "F₁ score"
        fig
    end
end

# test
function test_code()
    bins = range(-200.0, stop=50.0, length=500)
    idx1 = -35.0
    idx2 = 0.0
    r = 1.0 .- 0.7*1.0./(1 .+ exp.(-(bins .- idx1)/2.0)) .+ 0.7./(1.0 .+ exp.(-(bins .-idx2)/4.0))
    θ = repeat(range(0.0, stop=2π, length=20),1,1)
    rr = permutedims(repeat(r,1,1))
    # construct the Manifold
    X = permutedims(repeat(bins, 1,20))
    Z = rr.*cos.(θ)
    Y = rr.*sin.(θ)
    ∇(x,y,z) = nothing
    # now draw paths on this Manifold
    fig,ax = lines(bins, r)
    vlines!(ax, [idx1, idx2])
    ax3 = Axis3(fig[2,1])
    surface!(ax3, X,Y,Z)
    fig
end
end