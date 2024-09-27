module TimeWindowDecoding
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
using JLD2
using CRC32c
using ProgressMeter

include("utils.jl")

plot_theme = theme_minimal()
plot_theme.Axis.xticksvisible[] = true
plot_theme.Axis.yticksvisible[] = true
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
    return lda, Xc
end

function decode_window(ppsth, trialidx::Vector{Vector{Int64}}, tlabels::Vector{Vector{Int64}}, rtimes, window;area::String="FEF", locations::Union{Nothing, Vector{Int64}}=nothing, rtime_min=120.0, rtime_max=300.0, combine_locations=true,nruns=100, ntrain=1500,ntest=500)
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
    # keep (a subsample of the) training data
    Xc = fill(0.0, ncells, 200, nruns) 
    prog = Progress(nruns)
    Threads.@threads for r in 1:length(RNGs)
        Yt, train_label,test_label =  EventOnsetDecoding.sample_trials(permutedims(Xtot,[3,2,1]), label_tot;RNG=RNGs[r])
        ntrain,ntest = length.([train_label, test_label])
        Ytest = Yt[:, ntrain+1:end, :]
        Ytrain = Yt[:, 1:ntrain, :]
        # train LDA
        lda,_Xc = decode_window(Ytrain, bins, window)
        Xc[:, 1:100,r] = _Xc[:,shuffle(1:ntrain)[1:100]]
        Xc[:, 101:200,r] = _Xc[:,shuffle(ntrain+1:2*ntrain)[1:100]]
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
        next!(prog)
   end
   finish!(prog)
   posterior,bins[1:size(posterior,1)], Xc
end

function plot(bins::AbstractVector{Float64}, posterior::Matrix{Float64},window::Tuple{Float64, Float64};kvs...)
    fig = Figure()
    ax = Axis(fig[1,1])
    plot!(ax, bins, posterior,window)
    ax.xlabel = "Time [ms]"
    ax.ylabel = "Performance"
    fig
end

function plot!(ax, bins::AbstractVector{Float64}, posterior::Matrix{Float64},window::Tuple{Float64, Float64};kvs...)
    vspan!(ax, window[1], window[2],color=RGB(0.8, 0.8, 0.8))
    μ = dropdims(mean(posterior, dims=2), dims=2)
    σ = dropdims(std(posterior, dims=2), dims=2)
    fill_between!(ax, bins, μ-σ , μ+σ)
    lines!(ax, bins, μ)
    ax
end

function plot(bins, posteriors, windows::Vector{Tuple{Float64, Float64}})
    n = length(posteriors)
    with_theme(theme_minimal()) do
        fig = Figure(resolution=(400,500))
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
        ax.ylabel = "Performance"
        fig
    end
end

plot(;kvs...) = plot([(-35.0, 0.0), (-70.0, -35.0), (-105.0, -70.0), (-140.0, -105.0)];kvs... )
plot2(;kvs...) = plot2([(-35.0, 0.0), (-70.0, -35.0), (-105.0, -70.0), (-140.0, -105.0)];kvs... )

function plot(windows::Vector{Tuple{Float64, Float64}};q0=zero(UInt32), kvs...)
    q = CRC32c.crc32c(string(windows),q0)
    h = string(q, base=16)
    fname = "time_window_decoding_$h.jld2"
    fname = joinpath("data",fname)
    if isfile(fname)
        bins, posteriors,windows = JLD2.load(fname, "bins", "posteriors","windows")
    else
        ppsth,tlabels,trialidx, rtimes = JLD2.load("data/ppsth_mov.jld2", "ppsth","labels","trialidx","rtimes")
        posteriors = Any[]
        bins = ppsth.bins
        for window in windows
            posterior,bins = decode_window(ppsth,trialidx,tlabels, rtimes, window;nruns=100)
            push!(posteriors, posterior)
        end
        JLD2.save(fname, Dict("bins"=>bins, "posteriors"=>posteriors, "windows"=>windows))
    end
    fig = plot(bins, posteriors, windows)
end

function plot2(windows::Vector{Tuple{Float64, Float64}};q0=zero(UInt32), width=300,height=200, kvs...)
    q = CRC32c.crc32c(string(windows),q0)
    h = string(q, base=16)
    fname = "time_window_decoding_$h.jld2"
    fname = joinpath("data",fname)
    @show fname
    if isfile(fname)
        bins, posteriors,windows,Xp = JLD2.load(fname, "bins", "posteriors","windows","Xp")
    else
        ppsth,tlabels,trialidx, rtimes = JLD2.load("data/ppsth_mov.jld2", "ppsth","labels","trialidx","rtimes")
        posteriors = Any[]
        Xp = Any[]
        bins = ppsth.bins
        for window in windows
            posterior,bins,Xc = decode_window(ppsth,trialidx,tlabels, rtimes, window;nruns=100)
            push!(posteriors, posterior)
            push!(Xp, Xc)
        end
        JLD2.save(fname, Dict("bins"=>bins, "posteriors"=>posteriors, "Xp"=>Xp, "windows"=>windows))
    end
    with_theme(plot_theme) do
        fig = Figure(resolution=(width,height))
        ax = Axis(fig[1,1])
        for (ii,(window,posterior)) in enumerate(zip(windows,posteriors))
            μ = dropdims(mean(posterior, dims=2), dims=2)
            σ = dropdims(std(posterior, dims=2), dims=2)
            fill_between!(ax, bins, μ-σ , μ+σ)
            lines!(ax, bins, μ)
            vlines!(ax, window[1], color=Cycled(ii))
        end
        ax.xlabel = "Time from movement [ms]"
        ax.ylabel = "Performance"
        ax2 = Axis(fig[1,2])
        ll = [fill(1,100);fill(2,100)]
        μ = fill(0.0, 2, length(Xp))
        σ = fill(0.0, 2, length(Xp))
        for i in 1:length(Xp)
            lda = fit(SubspaceLDA, Xp[i][:,:,1], ll)
            Xv = predict(lda, Xp[i][:,:,1])
            _σ = fill(0.0, 2, size(Xp[i],3))
            _μ = fill(0.0, 2, size(Xp[i],3))
            x1 = Float64[]
            x2 = Float64[]
            window = windows[i]
            w = window[2]-window[1]
            for r in 1:size(Xp[1],3)
                lda = fit(SubspaceLDA, Xp[i][:,:,r], ll)
                x = predict(lda, Xp[i][:,:,r])
                _σ[1,r] = std(x[1:100])
                _μ[1,r] = mean(x[1:100])
                _σ[2,r] = std(x[101:200])
                _μ[2,r] = mean(x[101:200])
                append!(x1, filter(_x->abs(_x-_μ[2,r])<3*_σ[1,r], x[1:100]) .- _μ[1,r])
                append!(x2, filter(_x->abs(_x-_μ[1,r])<3*_σ[2,r], x[101:200]) .- _μ[2,r])
            end
            #μ[:,i] = [mean(Xv[1:100]); mean(Xv[101:200])]
            μ[:,i] = dropdims(mean(_μ,dims=2),dims=2)
            σ[:,i] = dropdims(mean(_σ,dims=2),dims=2)
            #scatter!(ax2, (i-1) .+ 0.8*rand(size(Xv,2)), Xv[1,:],markersize=10px)
            color = Makie.wong_colors()[i]
            scatter!(ax2, window[1]-0.5*w .+ 0.8*w*rand(length(x1)), x1 .+ μ[1,i],markersize=8px,color=color)
            scatter!(ax2, window[1]-0.5*w .+ 0.8*w*rand(length(x2)), x2 .+ μ[2,i],markersize=8px,color=color)
            errorbars!(ax2, [window[1]], [μ[1,i]], [σ[1,i]],color="black")
            errorbars!(ax2, [window[1]], [μ[2,i]], [σ[2,i]],color="black")
            scatter!(ax2, fill(window[1],2), μ[:,i], color=color)
            ax2.yticklabelsvisible = false
            ax2.xticks = [w[1] for w in windows]
            ax2.ylabel = "LDA projection [au]"
        end

        colsize!(fig.layout, 1, Relative(0.6))
        colgap!(fig.layout, 1, 5)
        labels = [Label(fig[1,1,TopLeft()], "A"),
                  Label(fig[1,2,TopLeft()], "B")]
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