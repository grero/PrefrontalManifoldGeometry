module TrajectoryEstimation
using MultivariateStats
using Makie
using GLMakie
using StatsBase
using DataProcessingHierarchyTools
# for GPFA
using NeuralAnalysis
using PyCall
neocore = pyimport("neo.core")
quantities = pyimport("quantities")
GPFA = pyimport("elephant.gpfa")

const DPHT = DataProcessingHierarchyTools

include("utils.jl")

abstract type AbstractTimeReference end

struct ReactionTime <: AbstractTimeReference end
struct TrialEnd <: AbstractTimeReference end
struct TrialStart <: AbstractTimeReference end
struct TimePoint <: AbstractTimeReference
    t::Float64
end

function get_spiketrains(cellnames::Vector{String}, rargs::NeuralAnalysis.RasterArgs, datadir::String=pwd())
    spikes = Float64[]
    trialidx = Int64[]
    cellidx = Int64[]
    for (ii,cname) in enumerate(cellnames)
        raster = cd(joinpath(datadir, cname)) do
            NeuralAnalysis.get_raster(rargs)
        end
        append!(spikes, raster.raster.events)
        append!(trialidx, raster.raster.trialidx)
        append!(cellidx, fill(ii, length(raster.raster.events)))
    end
    spikes, trialidx, cellidx
end

function get_gpfa_trajectories(cellnames::Vector{String}, rargs::NeuralAnalysis.RasterArgs, datadir::String=pwd();binsize=20.0)
    spikes,trialidx,cellidx = get_spiketrains(cellnames, rargs, datadir)
    get_gpfa_trajectories(spikes,trialidx,cellidx,binsize;t_start=rargs.tmin,t_stop=rargs.tmax)
end

function get_gpfa_trajectories(spikes::Vector{Float64}, trialidx::Vector{Int64}, cellidx::Vector{Int64},binsize;latent_dims=3,
                                                                                                                t_start=minimum(spikes),
                                                                                                                t_stop=maximum(spikes))
    # convert to neo_spiketrains
    neo_spiketrains = [[pycall(neocore.spiketrain.SpikeTrain, PyObject, spikes[(cellidx.==i).&(trialidx.==j)], t_stop=t_stop, t_start=t_start, units=quantities.ms) for i in 1:maximum(cellidx)] for j in 1:maximum(trialidx)]
    gpfa = GPFA.GPFA(bin_size=binsize*quantities.ms, x_dim=latent_dims,svd_method="lapack",tau_init=50.0*quantities.ms)
    gpfa.fit(neo_spiketrains)
    gpfa, gpfa.transform(neo_spiketrains)
end

"""
    get_traj_index(_traj::Array{T,3},cutoff=2.0) where T <: Real

Return an index of trajectories with a mean mahalanobis distance from the mean less than or equal to `cutoff`
"""
function get_traj_index(_traj::Array{T,3},cutoff=2.0) where T <: Real
    d,nb,nt = size(_traj)
    dst = fill(0.0, nb,nt)
    for i in 1:nb
        μ = mean(_traj[:,i,:],dims=2)
        Σ = cov(_traj[:,i,:],dims=2)
        Zi = inv(Σ)
        for j in 1:nt
            dd = _traj[:,i,j] .- μ
            dq = dd'*Zi*dd
            dst[i,j] = sqrt(dq[1,1])
        end
    end
    dstm = dropdims(mean(dst, dims=1),dims=1)
    findall(dstm .< cutoff)
end
"""
    get_index(ref::AbstractTimeReference, bins::AbstractVector{T}, rt::T) where T <: Real

Return an index into `bins` based reference type.
"""
get_index(ref::AbstractTimeReference, bins::AbstractVector{T}, rt::T) where T <: Real = error("Not implemented")

get_index(ref::TrialStart, bins::AbstractVector{T}, rt::T) where T <: Real = 1
get_index(ref::TrialEnd, bins::AbstractVector{T}, rt::T) where T <: Real = length(bins) 

function get_index(ref::ReactionTime, bins::AbstractVector{T}, rt::T) where T <: Real
    searchsortedfirst(bins, rt)
end

function get_index(ref::TimePoint, bins::AbstractVector{T}, rt::T) where T <: Real
    searchsortedfirst(bins, ref.t)
end

"""
Create trajectory in 3D space by creating a subspace using the window defined by 
`t0` and `t1`. Then, project trials onto this subspace.

Since we want these trajectories to correlate with reaction time, we use Factor Analysis, which allows
us to create a space representing the shared variance among the neurons
"""
function get_trajectories(ppsth, labels,trialidx, rtimes, subject::String;projection_type=FactorAnalysis, t0::AbstractTimeReference=TimePoint(-50.0), t1::AbstractTimeReference=TrialEnd(),rt_percentile_bin=5.0, window::Union{Nothing,Float64}=nothing,kvs...)
    bins = ppsth.bins
    nbins = length(bins)

    all_sessions = DPHT.get_level_path.("session", ppsth.cellnames)
    subjects = DPHT.get_level_name.("subject", ppsth.cellnames)
    sidx = findall(subjects.==subject)
    sessions = unique(all_sessions[sidx])
    ncells = fill(0, length(sessions))
    Y = fill(0.0, 3, size(ppsth.counts,1),sum(length.(labels)))
    rtlabel = fill(0, size(Y,3))
    reaction_time = fill(0.0, length(rtlabel))
    rtbins = range(0.0, step=rt_percentile_bin, stop=100.0)
    offset = 0
    nt = fill(0, length(sessions))
    for (sessionidx,session) in enumerate(sessions)
		X, _labels, _rtimes = Utils.get_session_data(session, ppsth, trialidx, labels, rtimes;mean_subtract=true, variance_stabilize=true,kvs...)
        nt[sessionidx] = size(X,2)
        _nt = nt[sessionidx]
		ncells[sessionidx] = size(X,3)

        #build space before movement onset
        Xtr = fill(0.0, size(X,3), _nt*nbins)
        idx0 = 1
        _offset = 0
        for i in 1:_nt
            idx0 = get_index(t0, bins, _rtimes[i])
            idx1 = get_index(t1, bins, _rtimes[i])
            #Xtr[:,i] = permutedims(dropdims(mean(X[idx0:idx1,i,:],dims=1),dims=1))
            Xtr[:,_offset+1:_offset+idx1-idx0+1] = permutedims(X[idx0:idx1,i,:])
            _offset += idx1-idx0+1
        end
        Xtr = Xtr[:,1:_offset]
        #Xtr = dropdims(mean(X[idx0:idx1,:,:],dims=1),dims=1)
        fa = fit(projection_type, Xtr;maxoutdim=3)
        if window !== nothing 
            X2,b2 = Utils.rebin2(X, bins, window)
        else
            X2 = X
        end
        # classify trials by reaction time
        sidx = sortperm(_rtimes)
        qidx = 99*(sidx .- minimum(sidx))./(maximum(sidx)-minimum(sidx)) .+ 1 
        # loop over trials
        for i in 1:_nt
            Y[:,:,offset+i] = predict(fa, permutedims(X2[:,i,:]))
            rtlabel[offset+i] = searchsortedlast(rtbins, qidx[i])
        end
        reaction_time[offset+1:offset+_nt] .= _rtimes
        offset += _nt
    end
    Y[:,:,1:offset], rtlabel[1:offset], reaction_time[1:offset], nt
end

function plot_trajectories(trajA::Array{T,3}, bins, rt, labels;events::Dict=Dict(), indicate_movement_onset=true, indicate_end=true) where T <: Real
    ulabels = unique(labels)
    sort!(ulabels)
    nlabel = length(ulabels)
    qq = [0.0, percentile(rt, [25, 50,75])..., maximum(rt)]
    lw = [1.0, 2.0, 3.0, 4.0]
    fig = Figure(); ax = Axis3(fig[1,1])
    idx0 = searchsortedfirst(bins, 0.0)
    colors = Makie.wong_colors()
    idxe = size(trajA,2)
    for l in ulabels
        μ = fill(0.0, 3, size(trajA,2), nlabel)
        for (ii,(_l, _m)) in enumerate(zip(qq[1:end-1], qq[2:end]))
            tidx = findall((labels.==l).&(_l .<= rt .<= _m))
            _rt = mean(rt[tidx])
            idxx = [idx0]
            icolors = [:black]
            if indicate_movement_onset
                idxt = searchsortedfirst(bins, _rt)
                push!(idxx, idxt)
                push!(icolors, :red)
            end
            if indicate_end 
                push!(idxx, idxe)
                push!(icolors, :grey)
            end
            μ[:,:,ii] = dropdims(mean(trajA[:,:,tidx], dims=3),dims=3)
            lines!(ax, μ[1,:,ii], μ[2,:,ii], μ[3,:,ii],color=colors[l],linewidth=lw[ii])
            scatter!(ax, μ[1,idxx,ii], μ[2,idxx,ii], μ[3, idxx,ii], color=icolors)
        end
    end
    # create legends for everything
    group_width = [LineElement(linewidth=_lw, color=:black) for _lw in lw]
    group_color = [PolyElement(color=_color) for _color in colors[1:length(ulabels)]]
    legend = Legend(fig, [group_width, group_color], [["<25th","25th-50th","50th-75th",">75th"],
                                                   ["Loc$i" for i in 1:4]],
                                                    ["rtime","location"])

    fig[1,2] = legend
                                                   
    fig
end
end # module