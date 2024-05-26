module Figure3
using MultivariateStats
using JLD2
using LinearRegressionUtils
using Distributions
using CairoMakie
using Makie: Colors
using Colors
using Random
using CRC32c
using LinearAlgebra
using ProgressMeter
using HypothesisTests
using StatsBase
using Printf
using DrWatson

using ..Utils
using ..PlotUtils
using ..Trajectories

"""
    get_projected_energy(X, bins, rt)

Get the maximum square deviation from a straigth path from the beginning to the end
of the transition period.
"""
function get_projected_energy(X::Array{T,3}, qidx::Vector{T2}, label::AbstractVector{Int64}=fill(1,size(X,2))) where T <: Real where T2 <: AbstractVector{Int64}
    nbins,ntrials,ncells = size(X)
    nn = countmap(label)
    ulabel = collect(keys(nn))
    nlabel = length(ulabel)
    sort!(ulabel)

    μ0 = fill(0.0, ncells, nlabel)
    μ1 = fill(0.0, ncells, nlabel)
    V = fill(0.0, ncells,nlabel)
    for j2 in 1:ntrials
        l = findfirst(label[j2].==ulabel)
        idxq,idxs = qidx[j2]
        Xs = X[idxq:idxs,j2,:]
        μ0[:,l] .+= Xs[1,:]
        μ1[:,l] .+= Xs[end,:]
    end
    for (kk,l) in enumerate(ulabel)
        _n = nn[l]
        μ0[:,kk] ./= _n
        μ1[:,kk] ./= _n
        V[:,kk] = μ1[:,kk] - μ0[:,kk]
        V[:,kk] ./= norm(V[:,kk])
    end
    get_projected_energy(X,qidx,V,label),V
end

function get_projected_energy(X::Array{T,3}, qidx::Vector{T2}, V::Matrix{Float64}, label::AbstractVector{Int64}=fill(1,size(X,2))) where T <: Real where T2 <: AbstractVector{Int64}
    nbins,ntrials,ncells = size(X)
    MM = fill(0.0, ntrials)
    nn = countmap(label)
    ulabel = collect(keys(nn))
    nlabel = length(ulabel)
    sort!(ulabel)
    for j2 in 1:ntrials
        l = findfirst(ulabel.==label[j2])
        #Xs = X[idxq:idxs,j2,:]
        #MM[j2] = maximum(sum(abs2, Xs .- (Xs*V).*permutedims(V),dims=2))
        MM[j2] = get_projected_energy(X[:,j2,:],qidx[j2],V[:,l])
    end 
    MM
end

function get_projected_energy(X::Matrix{Float64}, qidx::AbstractVector{Int64},V)
    Xs = X[qidx[1]:qidx[2],:]
    get_projected_energy(Xs, V)
end

function get_projected_energy(Xs::Matrix{T},V::Vector{T}) where T <: Real
    maximum(sum(abs2, Xs .- (Xs*V).*permutedims(V),dims=2)) 
end

function balance_num_trials(label::Vector{T}, args...) where T
    cc = countmap(label)
    nn,_ = findmin(cc)
    idxt = fill(0, nn*length(cc))
    for (i,l) in enumerate(sort(collect(keys(cc))))
        idx = findall(ll->ll==l, label)
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

function get_regression_data(subject;area="fef", align=:cue, raw=false, kvs...)
    ppsth,tlabels,trialidx, rtimes = load_data(subject;area=area,align=align,raw=raw)
    get_regression_data(ppsth,tlabels,trialidx,rtimes,subject;kvs...)
end

function load_data(subject::Union{String,Nothing}=nothing;area="fef",align=:cue, raw=false,kvs...)
    if subject == "M"
        # this is model data
        fname = joinpath("data","ppsth_model_cue.jld2")
    elseif raw
        fname = joinpath("data","ppsth_$(area)_$(align)_raw.jld2")
    else 
        fname = joinpath("data","ppsth_$(area)_$(align).jld2")
    end
    ppsth,tlabels,trialidx, rtimes = JLD2.load(fname, "ppsth","labels","trialidx","rtimes")
    return ppsth,tlabels,trialidx,rtimes
end

"""
    get_regression_data(subject;area="fef", rtmin=120.0, rtmax=300.0, window=35.0, Δt=15.0,align=:mov, realign=true, raw=false, do_shuffle=false, nruns=100,smooth_window::Union{Nothing, Float64}=nothing, kvs...)

Get data for regressing reaction time for each point in time for the specified `subject` and `area`.
"""
function get_regression_data(ppsth,tlabels,trialidx,rtimes,subject::Union{Nothing,String};rtmin=120.0, rtmax=300.0, window=35.0, Δt=15.0,t1=35.0, realign=true, do_shuffle=false, do_shuffle_responses=false, do_shuffle_time=false, do_shuffle_trials=false, shuffle_within_locations=false, nruns=100,smooth_window::Union{Nothing, Float64}=nothing, use_midpoint=false,tt=65.0, use_log=false, subtract_location_mean=false, use_new_energy_point_algo=false, tmin=-Inf, tmax=Inf, kvs...)

	# Per session, per target regression, combine across to compute rv
	all_sessions = Utils.DPHT.get_level_path.("session", ppsth.cellnames)
    subjects = Utils.DPHT.get_level_name.("subject", ppsth.cellnames)
    if subject !== nothing
        cidx = subjects .== subject
    else
        cidx = 1:length(all_sessions)
    end
	sessions = unique(all_sessions[cidx])
	rtp = fill(0.0, 0, size(ppsth.counts,1))
	rt = fill(0.0, 0)
    label = fill(0,0)
    ncells = fill(0,0)
    sessionidx = fill(0,0)
	bins = ppsth.bins
    nbins = length(bins)
	rtime_all = Float64[]
    r2j = fill(0.0, size(ppsth.counts,1),100)
    total_idx = 1
    n_tot_trials =  sum([length(rtimes[k]) for k in sessions])
    start_idx = searchsortedfirst(ppsth.bins, tmin)
    end_idx = searchsortedlast(ppsth.bins, tmax)
    Z = fill(0.0,n_tot_trials, end_idx-start_idx+1)
    L = fill!(similar(Z),NaN) 
    FL = fill!(similar(Z),NaN) 
    EE = fill!(similar(Z),NaN)
    MM = fill!(similar(Z),NaN) # modified enerngy
    SS = fill!(similar(Z), NaN) # speed
    SM = fill!(similar(Z), NaN) # speed
    offset = 0
    idxsm = 0 
    nnmax = 0
    warnfirst = true
	for (ii, session) in enumerate(sessions)
		X, _label, _rtime = Utils.get_session_data(session,ppsth, trialidx, tlabels, rtimes;rtime_min=rtmin,rtime_max=rtmax,kvs...)
        _ncells = size(X,3)
        nbins = size(X,1)
        _nt = length(_rtime)
        ulabel = unique(_label)
        sort!(ulabel)
        nlabel = length(ulabel)
        if do_shuffle_time
            for k1 in axes(X,2)
                bidx = shuffle(1:nbins)
                for k2 in axes(X,3)
                    X[:,k1,k2] = X[bidx,k1,k2]
                end
            end
        elseif do_shuffle_trials
            tridx = collect(1:_nt)
            if shuffle_within_locations
                for l in ulabel       
                    _tidx = findall(_label.==l)
                    tridx[_tidx] = shuffle(tridx[_tidx])
                end
            else
                shuffle!(tridx)
            end
            X = X[:,tridx,:]
        end
        nn = countmap(_label)
        if use_log
            _lrt = log.(_rtime)
        else
            _lrt = _rtime
        end
        #if smooth_window !== nothing
        #    X2 = mapslices(x->Utils.gaussian_smooth(x,bins,smooth_window), X, dims=1)
        #else
        #    X2 = X
        #end
        qidx = Vector{UnitRange{Int64}}(undef, _nt)
        idx0 = 1
        idx1 = searchsortedfirst(bins, bins[idx0]+window)
        idxq = searchsortedfirst(bins, bins[idx1]+Δt)
        # length of transition period, from window + Δt
        for j2 in 1:_nt
            idxs = searchsortedlast(bins, bins[idx0]+_rtime[j2]-t1)
            qidx[j2] = 1:(idxs-idxq+1)
            nnmax = max(nnmax, length(qidx[j2]))
        end
        if do_shuffle
            # shuffle per location
            qidxs = Vector{Vector{Int64}}(undef, _nt)
            for l in ulabel
                _tidx = findall(_label.==l)
                qidxs[_tidx] = shuffle_reaction_times(qidx[_tidx])
            end
        elseif do_shuffle_responses
            qidxs = shuffle(qidx)
        else
            qidxs = qidx
        end
        # compute M and V for the actual transition period
        idxt = searchsortedfirst(bins, tt) 
        idxp = searchsortedfirst(bins[start_idx:end_idx], tt-Δt-window)
        MM[offset+1:offset + _nt,idxp-1], V = get_projected_energy(X, map(x->x .+ (idxt-1), qidxs),_label)
        Nv = fill(0.0, size(V,1), size(V,1)-1, size(V,2))
        for kj in axes(V,2)
            Nv[:,:,kj] = nullspace(adjoint(V[:,kj]))
        end
        for (jj,j) in enumerate(start_idx:end_idx)
            idx0 = j
            idx1 = searchsortedlast(bins, bins[j]+window)
            idxq = searchsortedlast(bins, bins[idx1]+Δt)
            # project onto FA components
            y = permutedims(dropdims(sum(X[idx0:idx1,:,:],dims=1),dims=1))
            try
                fa = StatsBase.fit(MultivariateStats.FactorAnalysis,y;maxoutdim=1, method=:em)
                z =  permutedims(predict(fa, y))
                if realign
                    # check the sign of the correlation between reaction time and activity and realign so that the relationship is positive
                    β = llsq(z, _lrt)
                    if β[1] < 0.0
                        z .*= -1.0
                    end
                end
                Z[offset+1:offset+_nt,jj]  .= z
            catch ee
                @warn "Error encountered in FA for bin $j session $session"
            end
            # compute the average state at the beginning and at the end of the transition period
            # path length
            # shift the transition period
            _qidx = map(x->x .+ (idxq-1), qidxs)
            for j2 in 1:_nt
                l = findfirst(_label[j2].==ulabel)
                _idxv = _qidx[j2]
                if last(_idxv) >= length(bins)
                    continue
                end
                # transition period
                Xs = X[_idxv,j2,:]
                _bins = bins[_idxv]
                if use_midpoint
                    # midpoint
                    ip = div(length(_idxv),2)+1
                else
                    # point of high energy
                    #ee = dropdims(sum(abs2, Xs .- Xs[1:1,:],dims=2),dims=2)
                    ee = dropdims(sum(abs2, Xs,dims=2),dims=2)
                    if use_new_energy_point_algo
                        #ip = Trajectories.get_maximum_enenrgy_point(Xs, V[:,l])
                        ip = Trajectories.get_maximum_energy_point(Xs, Nv[:,:,l])
                    else
                        ip = argmax(ee)
                    end
                    if ip == 0
                        continue
                    end
                end
                δt = [_bins[ip]-_bins[1], _bins[end]-_bins[ip]]
                if smooth_window !== nothing
                    Xs = mapslices(x->Utils.gaussian_smooth(x, bins[_idxv], smooth_window), Xs, dims=1)
                end
                L[offset+j2,jj] = Trajectories.compute_triangular_path_length(Xs, ip)
                s1 = sqrt(sum(abs2,(Xs[1,:]-Xs[ip,:])./δt[1]))
                s2 = sqrt(sum(abs2, (Xs[end,:]-Xs[ip,:])./δt[2]) )
                ss = [s1,s2]
                SM[offset+j2,jj] = mean(filter(isfinite, ss))
                # energy
                EE[offset+j2,jj] = maximum(sum(abs2, Xs .- Xs[1:1,:],dims=2))
                # modified energy; projected onto the line
                MM[offset+j2,jj] = get_projected_energy(Xs, V[:,l])
                # avg speed
                SS[offset+j2,jj] = mean(sqrt.(sum(abs2,diff(Xs,dims=1),dims=2)))
                FL[offset+j2,jj] = sum(sqrt.(sum(abs2,diff(Xs,dims=1),dims=2)))


            end
        end
        append!(rt, _lrt)
        append!(label, _label)
        append!(ncells, fill(size(X,3), length(_label)))
        append!(sessionidx, fill(ii, length(_label)))
        offset += _nt
    end
    Z[1:offset,:], L[1:offset,:], EE[1:offset,:,], MM[1:offset,:], SS[1:offset,:], FL[1:offset,:], SM[1:offset,:], rt, label, ncells, bins[start_idx:end_idx], sessionidx
end

function compute_regression(nruns::Int64, args...;kvs...)
    nt = size(args[1],1)
    trialidx = fill(0, nt, nruns) 
    for i in 1:nruns
        trialidx[:,i] = rand(1:nt,nt)
    end
    compute_regression(trialidx,args...;kvs...)...,trialidx
end 

function compute_regression(trialidx::Matrix{Int64}, args...;kvs...)
    nruns = size(trialidx,2)
    nt = size(args[1],1)
    tidx = rand(1:nt,nt)
    _β,_Δβ, _pv, _r²,varidx = compute_regression(args...;trialidx=trialidx[:,1],kvs...)
    nvars,nbins = size(_β)
    β = fill(0.0, nvars,nbins,nruns)
    Δβ = fill(0.0, nvars-1,nbins,nruns)
    pv = fill(0.0, nbins,nruns)
    r² = fill(0.0, nbins, nruns)
    β[:,:,1] = _β
    Δβ[:,:,1] = _Δβ
    pv[:,1] = _pv
    r²[:,1] = _r² 
    for i in 2:nruns
        tidx = rand(1:nt,nt)
        β[:,:,i],Δβ[:,:,i],pv[:,i],r²[:,i],_,_ = compute_regression(args...;trialidx=trialidx[:,i],kvs...)
    end
    β,Δβ, pv, r²,varidx
end

# TODO: Does this work?
function shuffle_reaction_times(qidx::Vector{UnitRange{Int64}})
    ntrials = length(qidx)
    nn = length.(qidx)
    qidxs = Vector{Vector{Int64}}(undef, ntrials)
    for i in 1:ntrials
        k = 0
        # TODO: What to do with the largest nn? there will be no trials fulfilling the below
        # Maybe just leave it unshuffled
        for j in shuffle(1:ntrials)
            if i == j
                continue
            end
            if nn[j] >= nn[i]
                k = j
                break
            end
        end
        if k > 0
            bidx = shuffle(qidx[k])[1:nn[i]]
            sort!(bidx)
            qidxs[i] = bidx
        else
            qidxs[i] = qidx[i]
        end
    end
    qidxs
end

function compute_regression(rt::AbstractVector{Float64}, L::Vector{Float64},args...;kvs...)
    compute_regression(rt, repeat(L,1,1),args...;kvs...)
end

"""
Compute
"""
function compute_regression(rt::AbstractVector{Float64}, L::Matrix{Float64}, args...;trialidx=1:size(L,1),shuffle_trials=false,exclude_pairs::Vector{Tuple{Int64,Int64}}=Tuple{Int64,Int64}[],save_all_β=false, kvs...)
    nbins = size(L,2)
    nvars = length(args)+1
    if save_all_β
        nvars += div(nvars*(nvars-1),2) - length(exclude_pairs)
    end
    β = fill(NaN, nvars+1, nbins)
    Δβ = fill(NaN, nvars, nbins)
    pv = fill(NaN, nbins)
    r² = fill(NaN, nbins)
    rss = fill(NaN, nbins)
    if shuffle_trials
        sidx = shuffle(trialidx)
    else
        sidx = trialidx
    end
    varidx = nothing
    for i in axes(L,2) 
        _tidx = 1:length(trialidx)
        do_skip = false
        for Z in [L,args...]
            if ndims(Z)==2
                _tidx = intersect(_tidx,findall(isfinite, Z[trialidx,i]))
                if isempty(_tidx)
                    do_skip = true
                    break
                end
            end
        end
        if do_skip
            continue
        end 
        # run two regression models; one without path length L and one with
        if length(args) > 0
            X_no_L = hcat([ndims(Z)== 2 ? Z[trialidx[_tidx],i] : Z[trialidx[_tidx]] for Z in args]...)
            if size(X_no_L,1) < 10
                continue
            end
            _exclude_pairs = [(k-1,j-1) for (k,j) in exclude_pairs]
            lreg_no_L = LinearRegressionUtils.llsq_stats(X_no_L, rt[sidx[_tidx]];do_interactions=true,exclude_pairs=_exclude_pairs, kvs...)
            X_with_L = [L[trialidx[_tidx],i] X_no_L]
            p1 = length(lreg_no_L.β)
            rss1 = lreg_no_L.rss
        else
            X_with_L = L[trialidx[_tidx],i:i] 
            p1 = 1
            rss1 = sum(abs2, rt[sidx[_tidx]] .- mean(rt[sidx[_tidx]]))
        end


        #lreg_no_L = LinearRegressionUtils.llsq_stats(X_no_L, rt[sidx];do_interactions=true, exclude_pairs=[(2,3)])
        #lreg_with_L = LinearRegressionUtils.llsq_stats(X_with_L, rt[sidx];do_interactions=true, exclude_pairs=[(2,3)])
        # subtract one from the exclude pair index
        if size(X_with_L,1) < 10
            continue
        end
        try
            lreg_with_L = LinearRegressionUtils.llsq_stats(X_with_L, rt[sidx[_tidx]];do_interactions=true,exclude_pairs=exclude_pairs, kvs...)

            # compute the F-stat for whether adding the path length results in a significantly better fit
            n = length(rt)
            rss2 = lreg_with_L.rss
            rss[i] = rss2
            p2 = length(lreg_with_L.β)
            F = (rss1 - rss2)/(p2-p1)
            F /= rss2/(n-p2)
            if p2 > p1
                pv[i] = 1.0 - cdf(FDist(p2-p1, n-p2), F)
            else
                pv[i] = lreg_with_L.pv
            end
            r²[i] = lreg_with_L.r²
            if save_all_β
                β[:,i] .= lreg_with_L.β
                Δβ[:,i] .= lreg_with_L.Δβ
                varidx = lreg_with_L.varidx
            else
                β[1,i] = lreg_with_L.β[1]
                Δβ[1,i] = lreg_with_L.Δβ[1]
                vidx = findall(ii->(length(ii)==2)&&((ii[1]==1)||(ii[2]==1)), lreg_with_L.varidx)
                β[2:end,i] = lreg_with_L.β[vidx]
                Δβ[2:end,i] = lreg_with_L.Δβ[vidx]
            end
        catch ee
            if !isa(ee, PosDefException) && !isa(ee, SingularException) && !isa(ee, DomainError)
                rethrow(ee)
            end
        end
    end
    β,Δβ, pv, r², rss, varidx
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

function compute_regression(;redo=false, varnames=[:L, :Z, :ncells, :xpos, :ypos], subjects=["J","W"], sessions::Union{Vector{Int64},Symbol}=:all, locations::Union{Symbol, Vector{Int64}}=:all, combine_subjects=false, tt=65.0,nruns=100, use_midpoint=false, shuffle_responses=false,shuffle_time=false, shuffle_trials=false, check_only=false, save_all_β=false, balance_positions=false, use_log=false, recording_side::Utils.RecordingSide=Utils.BothSides(),use_new_energy_point_algo=false, kvs...)
    # TODO: Add option for combining regression for both animals
    q = UInt32(0)
    input_args = Dict{String,Any}()
    if subjects != ["J","W"]
        q = crc32c(string((:subjects=>subjects)),q)
    end
    input_args["subjects"] = subjects
    if sessions != :all
        q = crc32c(string((:sessions=>sessions)),q)
    end
    q = crc32c(string(:varnames=>varnames),q)
    input_args["varnames"] = varnames
    q = crc32c(string((:use_midpoint=>use_midpoint)),q)
    input_args["use_midpoint"] = use_midpoint
    if shuffle_responses
        q = crc32c(string((:shuffle_responses=>true)),q)
    end
    input_args["shuffle_responses"] = shuffle_responses
    if shuffle_time
        q = crc32c(string((:shuffle_time=>true)),q)
    end
    input_args["shuffle_time"] = shuffle_time
    if shuffle_trials
        q = crc32c(string((:shuffle_trials=>true)),q)
    end
    input_args["shuffle_trials"] = shuffle_trials
    if combine_subjects
        q = crc32c(string((:combine_subjects=>true)),q)
    end
    input_args["combine_subjects"] = combine_subjects
    if nruns != 100
        q = crc32c(string((:nruns=>nruns)),q)
    end
    input_args["nruns"] = nruns
    if save_all_β
        q = crc32c(string((:save_all_β=>true)),q)
    end
    input_args["save_all_β"] = save_all_β
    if balance_positions
        q = crc32c(string((:balance_positions=>true)),q)
    end
    input_args["balance_positions"] = balance_positions
    if tmax < Inf
        q = crc32c(string((:tmax=>tmax)),q)
    end
    input_args["tmax"] = tmin
    if tmin > -Inf
        q = crc32c(string((:tmin=>tmin)),q)
    end
    input_args["tmin"] = tmin
    if use_log
        q = crc32c(string((:use_log=>true)),q)
    end
    input_args["use_log"] = use_log

    if use_residuals
        q = crc32c(string((:use_residuals=>true)),q)
    end
    input_args["use_residuals"] = use_residuals

    if !isa(recording_side, Utils.BothSides)
        q = crc32c(string((:recording_side=>Symbol(recording_side))),q)
    end
    input_args["recording_side"] = recording_side
    if locations !== :all
        q = crc32c(string(:locations=>locations),q)
    end
    input_args["locations"] = locations
    if use_new_energy_point_algo
        q = crc32c(string(:use_new_energy_point_algo=>true),q)
    end
    input_args["use_new_energy_point_algo"] = use_new_energy_point_algo 

    for k in kvs
        if !(k[1] == :t1 && k[2] == 0.0)
            q = crc32c(string(k),q)
        end
        input_args[string(k[1])] = k[2]
    end
    qs = string(q, base=16)
    fname = "path_length_regression_$(qs).jld2"
    if check_only
        @show fname
        return isfile(fname)
    end
    if isfile(fname) && redo == false
        qdata = JLD2.load(fname)
    else
        # apply git-tag
        tag!(input_args)
        qdata = Dict()
        bins = Float64[]
        for area in ["fef","dlpfc"]
            qdata[area] = Dict()
            Za = Vector{Matrix{Float64}}(undef, length(subjects))
            La = Vector{Matrix{Float64}}(undef, length(subjects))
            FLa = Vector{Matrix{Float64}}(undef, length(subjects))
            EEa = Vector{Matrix{Float64}}(undef, length(subjects))
            MMa = Vector{Matrix{Float64}}(undef, length(subjects))
            SSa = Vector{Matrix{Float64}}(undef, length(subjects))
            SMa = Vector{Matrix{Float64}}(undef, length(subjects))
            ncellsa = Vector{Vector{Float64}}(undef, length(subjects))
            xposa = Vector{Vector{Float64}}(undef, length(subjects))
            yposa = Vector{Vector{Float64}}(undef, length(subjects))
            lrta = Vector{Vector{Float64}}(undef, length(subjects))
            sessionidxa = Vector{Vector{Int64}}(undef, length(subjects))

            ppsth,tlabels,trialidx, rtimes = load_data(nothing;area=area,raw=true, kvs...)

            for (ss, subject) in enumerate(subjects)
                # load the data here so we don't have to do it more than once
                # TODO: Do the shuffling here (hic misce)
                Z,L,EE, MM, SS, FL, SM,lrt,label,ncells,bins,sessionidx = get_regression_data(ppsth,tlabels,trialidx, rtimes, subject; mean_subtract=true, variance_stabilize=true,window=50.0,use_midpoint=use_midpoint, use_log=use_log, use_new_energy_point_algo=use_new_energy_point_algo, tmin=tmin, tmax=tmax, kvs...);
                if subject == "J"
                    tidx = findall(label.!=9)
                else
                    tidx = 1:size(Z,1)
                end
                if sessions != :all
                    tidx = tidx[findall(in(sessions), sessionidx[tidx])]
                end
                if locations != :all
                    tidx = tidx[findall(in(locations), label[tidx])]
                end

                # use only trials associated with targets in the specfied hemifield
                tidx = tidx[findall(in(Utils.get_recording_side(recording_side, subject)), label[tidx])]
                xpos = [p[1] for p in Utils.location_position[subject]][label[tidx]]
                ypos = [p[2] for p in Utils.location_position[subject][label[tidx]]]
                Za[ss] = Z[tidx,:]
                La[ss] = L[tidx,:]
                FLa[ss] = FL[tidx,:]
                EEa[ss] = EE[tidx,:]
                MMa[ss] = MM[tidx,:]
                SSa[ss] = SS[tidx,:]
                SMa[ss] = SM[tidx,:]
                ncellsa[ss] = ncells[tidx]
                xposa[ss] = xpos
                yposa[ss] = ypos
                lrta[ss] = lrt[tidx]
                sessionidxa[ss] = sessionidx[tidx]
            end
            # concatenate across trials
            sessionidx = cat(sessionidxa...,dims=1)
            Z = cat(Za...,dims=1)
            L = cat(La...,dims=1)
            FL = cat(FLa...,dims=1)
            EE = cat(EEa..., dims=1)
            MM = cat(MMa..., dims=1)
            SS = cat(SSa..., dims=1)
            SM = cat(SMa..., dims=1)
            ncells = cat(ncellsa..., dims=1)
            xpos = cat(xposa..., dims=1)
            ypos = cat(yposa..., dims=1)
            lrt = cat(lrta..., dims=1)
            if length(lrt) == 0
                return qdata
            end
            # equalize positions if requested
            if balance_positions
                _,Z,L,EE,MM,SS,FL,SM, ncells,xpos,ypos,lrt = balance_num_trials(collect(zip(xpos,ypos)),Z,L,EE,MM,SS,FL,SM, ncells,xpos,ypos,lrt)
            end
            allvars = (Z=Z, L=L,EE=EE,MM=MM,SS=SS,FL=FL, SM=SM, ncells=ncells, xpos=xpos, ypos=ypos)
            # exclude ncells if we are only doing one session
            #vars = [lrt[tidx], L[tidx,:], Z[tidx,:], EE[tidx,:], MM[tidx,:], xpos, ypos] 
            vars = Any[lrt]
            use_varnames = Symbol[]
            for vv in varnames
                vk = allvars[vv]
                if length(unique(vk)) > 1
                    push!(vars,vk) 
                    push!(use_varnames, vv)
                end
            end
            qdata["varnames"] = use_varnames
            if all(in(use_varnames).([:xpos, :ypos]))
                exclude_pairs = [(findfirst(use_varnames.==:xpos),findfirst(use_varnames.==:ypos))]
            else
                exclude_pairs = Tuple{Int64, Int64}[]
            end
            if "trialix" in keys(qdata)
                _trialidx = qdata["trialidx"]
                βfef,Δβfef,pvfef,r²fef,varidx = compute_regression(_trialidx,vars...;exclude_pairs=exclude_pairs,save_all_β=save_all_β)
            else
                βfef,Δβfef,pvfef,r²fef,varidx, _trialidx = compute_regression(nruns,vars...;exclude_pairs=exclude_pairs, save_all_β=save_all_β)
                qdata["trialidx"] = _trialidx
            end
            qdata[area]["β"] = βfef
            qdata[area]["Δβ"] = Δβfef
            qdata[area]["pvalue"] = pvfef
            qdata[area]["r²"] = r²fef
            qdata[area]["varidx"] = varidx

            qdata["bins"] = bins
            qdata["sessionidx"] = sessionidx
            qdata["subjects"] = subjects
            # shuffle
            if shuffle_responses
                do_shuffle_responses = true
                do_shuffle = false
                do_shuffle_time = false
                do_shuffle_trials = false
            elseif shuffle_time
                do_shuffle_time = true
                do_shuffle = false
                do_shuffle_responses = false
                do_shuffle_trials = false
            elseif shuffle_trials
                do_shuffle_trials = true
                do_shuffle = false
                do_shuffle_responses = false
                do_shuffle_time = false
            else
                do_shuffle = true
                do_shuffle_responses = false
                do_shuffle_time = false
                do_shuffle_trials = false
            end

            # since we want to shuffle within subject, the other loop is over runs
            β = qdata[area]["β"]
            r² = qdata[area]["r²"]
            β_S = fill(0.0, size(β)...)
            r²_S = fill(0.0, size(r²)...)
            @showprogress for r in 1:nruns
                for (ss, subject) in enumerate(subjects)
                    _Z,_L,_EE, _MM, _SS, _FL, _SM, _lrt,_label,_ncells,bins,_sessionidx = get_regression_data(ppsth,tlabels,trialidx,rtimes,subject;mean_subtract=true, variance_stabilize=true,window=50.0,use_midpoint=use_midpoint, do_shuffle=do_shuffle, do_shuffle_responses=do_shuffle_responses,do_shuffle_time=do_shuffle_time,do_shuffle_trials=do_shuffle_trials,use_log=use_log, use_new_energy_point_algo=use_new_energy_point_algo,tmin=tmin, tmax=tmax, kvs...);
                    if subject == "J"
                        tidx = findall(_label.!=9)
                    else
                        tidx = 1:size(_Z,1)
                    end
                    if sessions != :all
                        tidx = tidx[findall(in(sessions), _sessionidx[tidx])]
                    end
                    tidx = tidx[findall(in(Utils.get_recording_side(recording_side, subject)), _label[tidx])]
                    _xpos = [p[1] for p in Utils.location_position[subject]][_label[tidx]]
                    _ypos = [p[2] for p in Utils.location_position[subject][_label[tidx]]]
                    Za[ss] = _Z[tidx,:]
                    La[ss] = _L[tidx,:]
                    FLa[ss] = _FL[tidx,:]
                    EEa[ss] = _EE[tidx,:]
                    MMa[ss] = _MM[tidx,:]
                    SSa[ss] = _SS[tidx,:]
                    SMa[ss] = _SM[tidx,:]
                    ncellsa[ss] = _ncells[tidx]
                    xposa[ss] = _xpos
                    yposa[ss] = _ypos
                    lrta[ss] = _lrt[tidx]
                end
                # need to again concatente
                Z = cat(Za...,dims=1)
                L = cat(La...,dims=1)
                FL = cat(FLa...,dims=1)
                EE = cat(EEa..., dims=1)
                MM = cat(MMa..., dims=1)
                SS = cat(SSa..., dims=1)
                SM = cat(SMa..., dims=1)
                ncells = cat(ncellsa..., dims=1)
                xpos = cat(xposa..., dims=1)
                ypos = cat(yposa..., dims=1)
                lrt = cat(lrta..., dims=1)
                if balance_positions
                    _,Z,L,EE,MM,SS,FL,SM, ncells,xpos,ypos,lrt = balance_num_trials(collect(zip(xpos,ypos)),Z,L,EE,MM,SS,FL,SM,ncells,xpos,ypos,lrt)
                end
                allvars = (Z=Z, L=L,EE=EE,MM=MM,SS=SS, FL=FL,SM=SM,ncells=ncells, xpos=xpos, ypos=ypos)

                vars = Any[lrt]
                for vv in use_varnames
                    push!(vars, allvars[vv])
                end
                if use_residuals
                    _β,_,_,_r²,_ = compute_regression(residuals, vars[2][qdata["trialidx"][:,r]];exclude_pairs=exclude_pairs,shuffle_trials=false,save_all_β=save_all_β, use_residuals=false)
                else
                    _β,_,_,_r²,_ = compute_regression(vars...;exclude_pairs=exclude_pairs,shuffle_trials=false,save_all_β=save_all_β, use_residuals=use_residuals, trialidx=qdata["trialidx"][:,r])
                end
                β_S[:,:,r] .= _β
                r²_S[:,r] .= _r²
            end
            qdata[area]["β_shuffle"]=β_S
            qdata[area]["r²_shuffled"]=r²_S
        end
        qdata["input_args"] = input_args
        JLD2.save(fname, qdata)
    end
    qdata
end 

function plot_fef_dlpfc_r²(;redo=false, subjects=["J","W"], sessions::Union{Vector{Int64},Symbol}=:all, tt=65.0,nruns=100, use_midpoint=false, kvs...)
    qdata = compute_regression(;redo=redo, subjects=subjects, sessions=sessions, tt=tt, nruns=100, use_midpoint=use_midpoint, kvs...)
    colors = [PlotUtils.fef_color, PlotUtils.dlpfc_color]
    fcolors = [RGB(0.5 + 0.5*c.r, 0.5+0.5*c.g, 0.5+0.5*c.b) for c in colors]
    with_theme(PlotUtils.plot_theme) do
        fig = Figure(size=(700,700))
        axes = [Axis(fig[i,1]) for i in 1:2*length(subjects)]
        for (ii,subject) in enumerate(subjects)
           
            bins = qdata[subject]["bins"]
            r²fef, βfef, βfef_S = [qdata[subject]["fef"][k] for k in ["r²","β","β_shuffle"]]
            r²dlpfc, βdlpfc, βdlpfc_S = [qdata[subject]["dlpfc"][k] for k in ["r²","β","β_shuffle"]]
            ax1 = axes[(ii-1)*2+1]
            @show size(r²fef)
            mid_fef = dropdims(median(r²fef,dims=2),dims=2)
            mid_dlpfc = dropdims(median(r²dlpfc,dims=2),dims=2)
            bidx = findall(isfinite, mid_fef)
            idxtt = searchsortedfirst(bins[bidx],tt)
            lower_fef,upper_fef = [mapslices(x->percentile(x,pc), r²fef[bidx,:],dims=2) for pc in [5,95]]
            lower_dlpfc,upper_dlpfc = [mapslices(x->percentile(x,pc), r²dlpfc[bidx,:],dims=2) for pc in [5,95]]
            band!(ax1, bins[bidx], lower_fef[:], upper_fef[:],color=(colors[1], 0.5))
            lines!(ax1, bins, mid_fef, label="FEF",color=PlotUtils.fef_color)
            band!(ax1, bins[bidx], lower_dlpfc[:], upper_dlpfc[:],color=(colors[2], 0.5))
            lines!(ax1, bins, mid_dlpfc, label="DLPFC",color=PlotUtils.dlpfc_color)

            ax1.title="Monkey $subject"
            ax1.ylabel = "r²"
            ax2 = axes[ii*2]
            ax2.ylabel = "β"
            lower_fef,mid_fef, upper_fef = [mapslices(x->percentile(x,pc), βfef[1,bidx,:],dims=2) for pc in [5,50,95]]
            lower_fef_S,mid_fef_S, upper_fef_S = [mapslices(x->percentile(x,pc), βfef_S[1,bidx,:],dims=2) for pc in [5,50,95]]
            lower_dlpfc,mid_dlpfc, upper_dlpfc = [mapslices(x->percentile(x,pc), βdlpfc[1,bidx,:],dims=2) for pc in [5,50,95]]
            lower_dlpfc_S,mid_dlpfc_S, upper_dlpfc_S = [mapslices(x->percentile(x,pc), βdlpfc_S[1,bidx,:],dims=2) for pc in [5,50,95]]

            band!(ax2, bins[bidx], lower_fef[:], upper_fef[:],color=(colors[1], 0.5))
            lines!(ax2, bins[bidx], mid_fef[:], label="FEF",color=colors[1])
            hlines!(ax2, mid_fef[idxtt],color=colors[1])
            band!(ax2, bins[bidx], lower_dlpfc[:], upper_dlpfc[:],color=(colors[2], 0.5))
            lines!(ax2, bins[bidx], mid_dlpfc[:], label="DLPFC",color=colors[2])
            hlines!(ax2, 0.0, color=:black, linestyle=:dot)
        end
        for ax in axes
            vlines!(ax, [0.0,tt], color=[:black,:gray])
        end
        linkaxes!(axes[1:2:end]...)
        linkxaxes!(axes...)
        for ax in axes[1:end-1]
            ax.xticklabelsvisible = false
        end
        ax = axes[end-1]
        axislegend(ax, halign=:left, margin=(5.0, 0.0, 0.0, -25.0))
        align = get(kvs, :align, :cue)
        if align == :cue
            axes[end].xlabel = "Time from go-cue"
        elseif align == :target
            axes[end].xlabel = "Time from target"
        end 

        fig
    end
end

function plot_path_length_regression_with_shuffles(;redo=false, subjects=["J","W"],nruns=100, tt=65.0,show_zscore=false, βindex=1,kvs...)
    # a bit of a hack; test if files have already been computed with individual subjects first
    if (subjects == ["J","W"] || subjects == ["W","J"]) && get(kvs, :combine_subjects, false) == false
        if !compute_regression(;subjects=subjects, tt=tt, nruns=nruns, check_only=true, kvs...)
            qdata = Dict()
            for subject in subjects
                _qdata = compute_regression(;redo=redo, subjects=[subject],tt=tt,nruns=nruns, kvs...)
                qdata[subject] = _qdata[subject]
            end
        end
    else 
        qdata = Dict()
        kk = join(subjects)
        _qdata = compute_regression(;redo=redo, subjects=subjects, nruns=nruns, tt=tt, kvs...)
        if kk in keys(_qdata)
            qdata = _qdata
        else
            qdata[kk] = _qdata
        end
    end
    plot_path_length_regression_with_shuffles(qdata;tt=tt,show_zscore=show_zscore,βindex=βindex)
end

function plot_path_length_regression_with_shuffles(qdata;tt=65.0,show_zscore=false, βindex=1, kvs...)
    @show βindex
    subjects = collect(keys(qdata))
    colors = [PlotUtils.fef_color, PlotUtils.dlpfc_color]
    fcolors = [RGB(0.5 + 0.5*c.r, 0.5+0.5*c.g, 0.5+0.5*c.b) for c in colors]
    with_theme(PlotUtils.plot_theme) do
        height = 100 + 200*length(subjects)
        fig = Figure(size=(500,height))
        if show_zscore
            axes = [Axis(fig[i,1]) for i in 1:length(subjects)]
        else
            axes = [Axis(fig[i,1]) for i in 1:2*length(subjects)]
        end
        for ax in axes
            vlines!(ax, [0.0, tt],color=[:black, :gray])
        end
        for (ii,subject) in enumerate(subjects)
            bins = qdata[subject]["bins"]
            r²fef, βfef, βfef_S = [qdata[subject]["fef"][k] for k in ["r²","β","β_shuffle"]]
            r²dlpfc, βdlpfc, βdlpfc_S = [qdata[subject]["dlpfc"][k] for k in ["r²","β","β_shuffle"]]
            bidx = findall(isfinite, dropdims(mean(r²fef,dims=2),dims=2))
            idxtt = searchsortedfirst(bins,tt)

            lower_fef,mid_fef, upper_fef = [mapslices(x->percentile(x,pc), βfef[βindex,bidx,:],dims=2) for pc in [5,50,95]]
            lower_fef_S,mid_fef_S, upper_fef_S = [mapslices(x->percentile(x,pc), βfef_S[βindex,bidx,:],dims=2) for pc in [5,50,95]]
            lower_dlpfc,mid_dlpfc, upper_dlpfc = [mapslices(x->percentile(x,pc), βdlpfc[βindex,bidx,:],dims=2) for pc in [5,50,95]]
            lower_dlpfc_S,mid_dlpfc_S, upper_dlpfc_S = [mapslices(x->percentile(x,pc), βdlpfc_S[βindex,bidx,:],dims=2) for pc in [5,50,95]]
            
            if show_zscore
                # compute a `z-score`, that is the mean of the actual data
                ax1 = axes[ii]
                ax1.title = "Monkey $(subject)"
                for (β,βs, color,area) in zip([βfef,βdlpfc],[βfef_S,βdlpfc_S],[PlotUtils.fef_color, PlotUtils.dlpfc_color],["FEF","DLPFC"])
                    @show size(βs) 
                    μ = dropdims(mean(β[βindex,bidx,:],dims=2),dims=2)
                    μs = dropdims(mean(βs[βindex,bidx,:],dims=2),dims=2)
                    σs = dropdims(std(βs[βindex,bidx,:],dims=2),dims=2)
                    zs = (μ - μs)./σs
                    lines!(ax1, bins[bidx], zs, label=area, color=color)
                    ax1.ylabel = "Zscored β"
                end
            else
                ax1 = axes[(ii-1)*2+1]
                ax2 = axes[ii*2]
                ax1.title = "Monkey $(subject)"
                band!(ax1, bins[bidx], lower_fef_S[:], upper_fef_S[:],color=(:black,0.5))
                lines!(ax1, bins[bidx], mid_fef_S[:], label="Shuffled",color=:black)

                band!(ax1, bins[bidx], lower_fef[:], upper_fef[:],color=(colors[1],0.5))
                lines!(ax1, bins[bidx], mid_fef[:], label="FEF",color=colors[1])

                band!(ax2, bins[bidx], lower_dlpfc_S[:], upper_dlpfc_S[:],color=(:black,0.5))
                lines!(ax2, bins[bidx], mid_dlpfc_S[:], label="Shuffled",color=:black)

                band!(ax2, bins[bidx], lower_dlpfc[:], upper_dlpfc[:],color=(colors[2],0.5))
                lines!(ax2, bins[bidx], mid_dlpfc[:], label="dlpfc",color=colors[2])
            end
        end
        if show_zscore
            linkaxes!(axes...)
            axislegend(axes[1], valign=:top, halign=:left,margin=(5.0, -5.0, -5.0, -30.0))
        else
            axislegend(axes[end-1], valign=:top, halign=:left,margin=(5.0, -5.0, -5.0, -30.0))
        end
        for ax in axes[1:end-1]
            ax.xticklabelsvisible = false
        end
        axes[end].xlabel = "Time from go-cue"
        fig
    end
end

function plot_β_comparison(qdata::Vector{T}, area::Vector{String}, βindex::Int64;kvs...) where T <: Dict
    with_theme(PlotUtils.plot_theme) do
        fig = Figure(size=(400,300))
        ax = Axis(fig[1,1])
        plot_β_comparison!(ax, qdata, area, βindex;kvs...)
        fig
    end
end

function plot_β_comparison!(ax, qdata::Vector{T}, area::Vector{String}, βindex::Int64;show_zscore=false, show_shuffled=false, tt=65.0, plot_r²=false,tmin=-Inf,tmax=Inf, labels::Union{Vector{String},Nothing}=nothing,fix_colors=true) where T <: Dict
    with_theme(PlotUtils.plot_theme) do
        vlines!(ax, [0.0, tt];color=[:black, (:gray, 0.5)])
        if plot_r²
            yl = "r²"
        else
            yl = "β"
        end
        kk = 1
        acolors = Dict{String,Int64}()
        for (ii,_qdata) in enumerate(qdata)
            bins = _qdata["bins"]
            tidx = searchsortedfirst(bins, tt)
            for _area in area
                _data = _qdata[_area]
                if plot_r²
                    Xs = _data["r²_shuffled"]
                    X = _data["r²"]
                else
                    Xs = _data["β_shuffle"][βindex,:,:]
                    X = _data["β"][βindex,:,:]
                end
                μ = dropdims(mean(X,dims=2),dims=2)
                μs = dropdims(mean(Xs,dims=2),dims=2)
                σs = dropdims(std(Xs,dims=2),dims=2)
                bidx = searchsortedfirst(bins, tmin):searchsortedlast(bins, tmax)
                bidx = intersect(bidx, findall(isfinite, σs))
                # kind of hacky
                la = lowercase(_area)
                if fix_colors
                    if la == "fef"
                        if la in keys(acolors)
                            acolors[la] += 1
                            _color = PlotUtils.fef_color_dark
                        else
                            acolors[la] = 1
                            _color = PlotUtils.fef_color
                        end
                    elseif la == "dlpfc"
                        if la in keys(acolors)
                            acolors[la] += 1
                            _color = PlotUtils.dlpfc_color_dark
                        else
                            acolors[la] = 1
                            _color = PlotUtils.dlpfc_color
                        end
                    else
                        _color = Cycled(kk) 
                    end
                else
                    _color = Cycled(kk) 
                end
                if show_shuffled
                    # indicate the 
                    band!(ax, bins[bidx], (μs-σs)[bidx], (μs+σs)[bidx],color=(_color, 0.5))
                    #μ = μs
                elseif show_zscore
                    μ .= (μ - μs)./σs
                end
                @show _qdata["subjects"] μ[tidx]
                if labels === nothing
                    label = label=join(_qdata["subjects"])
                else
                    label = labels[ii]
                end
                
                ll = lines!(ax, bins[bidx], μ[bidx],label="$(label) $_area", linewidth=1.5,color=_color)
                hlines!(ax, μ[tidx], linestyle=:dot, color=_color)
                kk += 1
            end
        end
        if show_zscore
            ax.ylabel = "Z-scored $yl"
        else
            ax.ylabel=yl
        end
        ax.xlabel = "Time from go-cue [ms]"
        axislegend(ax,valign=:bottom, halign=:right)
    end
end

function plot_model(qdata,args...;tt=0.0, correct_bias=true, kvs...)
    nruns = size(qdata["fef"]["r²"],2)
    ttidx = searchsortedfirst(qdata["bins"], 0.0)
    colors = [PlotUtils.fef_color, PlotUtils.dlpfc_color]
    with_theme(PlotUtils.plot_theme) do
        fig = Figure(size=(500,300))
        ax1 = Axis(fig[1,1])
        plot_β_comparison!(ax1, [qdata],args...;tt=tt,kvs...)
        ax2 = Axis(fig[1,2])
        xx = [fill(1, nruns);fill(2, nruns)]
        if correct_bias
            μ_fef = mean(qdata["fef"]["r²_shuffled"][ttidx,:])
            μ_dlpfc = mean(qdata["dlpfc"]["r²_shuffled"][ttidx,:])
            yy = [qdata["fef"]["r²"][ttidx,:] .- μ_fef;qdata["dlpfc"]["r²"][ttidx,:] .- μ_dlpfc]
        else
            yy = [qdata["fef"]["r²"][ttidx,:];qdata["dlpfc"]["r²"][ttidx,:]]
        end
        boxplot!(ax2, xx,yy ,
                    color=colors[xx])
        ax2.ylabel = "r²"
        ax2.xticklabelsvisible = false
        ax2.xticks = [1,2]
        colsize!(fig.layout, 1, Relative(0.7))
        @show MannWhitneyUTest(yy[xx.==1], yy[xx.==2])
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

function plot_β_summary(;subjects=["J","W"], kvs...)
    qdata = compute_regression(;subjects=subjects,kvs...)
    μ = Dict()
    σ² = Dict()
    for (ii,subject) in enumerate(subjects)
        bins, βfef, Δβfef,βdlpfc, Δβdlpfc = [qdata[subject][k] for k in ["bins","βfef","Δβfef","βdlpfc","Δβdlpfc"]]
        idx0 = searchsortedfirst(bins,-300.0)        
        idx1 = searchsortedfirst(bins, -150.0)
        idx2 = searchsortedfirst(bins, 65);
        # TODO: Use the standard deviations of the beta to estalish significance
        μ[subject] = Dict("dlpfc"=>Dict("transition"=>βdlpfc[1,idx2,:], "baseline"=>dropdims(mean(βdlpfc[1,idx0:idx1,:],dims=1),dims=1)),
                          "fef"=>Dict("transition"=>βfef[1,idx2,:], "baseline"=>dropdims(mean(βfef[1,idx0:idx1,:],dims=1),dims=1)))
    end

    colors = [PlotUtils.fef_color, PlotUtils.dlpfc_color]
    fcolors = [RGB(0.5 + 0.5*c.r, 0.5+0.5*c.g, 0.5+0.5*c.b) for c in colors]
    acolors = [colors;fcolors]
    with_theme(PlotUtils.plot_theme) do
        fig = Figure()
        ax1 = Axis(fig[1,1])
        Yt = cat([[μ[subject]["fef"]["transition"] μ[subject]["dlpfc"]["transition"]] for subject in subjects]...,dims=2)
        #Yt = [μ["J"]["fef"]["transition"] μ["J"]["dlpfc"]["transition"] μ["W"]["fef"]["transition"] μ["W"]["dlpfc"]["transition"]]
        #Yb = [μ["J"]["fef"]["baseline"] μ["J"]["dlpfc"]["baseline"] μ["W"]["fef"]["baseline"] μ["W"]["dlpfc"]["baseline"]]
        Yb = cat([[μ[subject]["fef"]["baseline"] μ[subject]["dlpfc"]["baseline"]] for subject in subjects]...,dims=2)

        lower_b,mid_b,upper_b = [dropdims(mapslices(x->percentile(x,pp),Yb,dims=1),dims=1) for pp in [5,50,95]]
        lower_t,mid_t,upper_t = [dropdims(mapslices(x->percentile(x,pp),Yt,dims=1),dims=1) for pp in [5,50,95]]
        #mid = [mid_t[1],mid_b[1],mid_t[2],mid_b[2],mid_t[3],mid_b[3],mid_t[4],mid_b[4]]
        mid = cat([[mid_t[i],mid_b[i]] for i in 1:length(mid_b)]...,dims=1)
        @show mid
        lower = cat([[lower_t[i],lower_b[i]] for i in 1:length(mid_b)]...,dims=1)
        upper = cat([[upper_t[i],upper_b[i]] for i in 1:length(mid_b)]...,dims=1)
        #lower = [lower_t[1],lower_b[1],lower_t[2],lower_b[2],lower_t[3],lower_b[3],lower_t[4],lower_b[4]]
        #upper = [upper_t[1],upper_b[1],upper_t[2],upper_b[2],upper_t[3],upper_b[3],upper_t[4],upper_b[4]]
        barplot!(ax1, 1:length(mid),mid,
                          color=acolors[cat([[1,3,2,4] for i in 1:length(subjects)]...,dims=1)])
       
        rangebars!(ax1, 1:length(mid), lower, upper)
        ax1.ylabel = "β"
        Legend(fig[1,1], [PolyElement(color=RGB(0.0, 0.0,0.0)),
                           PolyElement(color=RGB(0.5, 0.5, 0.5))],
                           ["Transition","Baseline"],tellwidth=false,
                           valign=:top)
        ax1.xticks = [2.5, 7.0]
        ax1.xticklabelsvisible = false
        ax = Axis(fig[2,1])

        # ratio
        func(x) = reduce(/, abs.(x))

        lower,mid,upper = [dropdims(mapslices(x->percentile(x,pp),abs.(Yt./Yb),dims=1),dims=1) for pp in [5,50,95]]

        barplot!(ax, 1:length(mid),mid,
                          color=colors[cat([[1,2] for i in 1:length(subjects)]...,dims=1)])

        rangebars!(ax, 1:length(mid), lower,upper)
        @show lower[1:2:end] upper[2:2:end], lower[2:2:end]
        Legend(fig[2,1], [PolyElement(color=colors[i]) for i in 1:2],["FEF", "DLPFC"],
                valign=:top, halign=:left, tellwidth=false)
        ax.ylabel = "β transition/β baseline"
        ax.xticks=(range(1.5, step=2.0, length=length(subjects)),["Monkey $subject" for subject in subjects])
        fig
    end
end

function plot(;plottype=:barplot, show_dlpfc=false)
    with_theme(PlotUtils.plot_theme) do
        fig = Figure(size=(700,600))
        # shematic
        lg1 = GridLayout(fig[1,1])
        plot_trajectory_illustration!(lg1)
        Label(lg1[1,1,TopLeft()], "A")
        lg2 = GridLayout(fig[2,1])
        plot_individual_trials!(lg2, "W";plot_joint=false, plottype=plottype, label=["B","C","D","E"],add_legend=false, xlabelvisible=false, shuffle_responses=false, shuffle_time=false, shuffle_trials=true, combine_subjects=true, save_all_β=true, shuffle_within_locations=true, t1=35.0, use_log=true)
        Label(lg2[1,0], "Monkey W", tellwidth=true, tellheight=false, rotation=π/2)
        lg21 = GridLayout(lg2[1,3])
        axw = plot_joint_regression!(lg21,"W";show_null_distr=false)

        lg3 = GridLayout(fig[3,1])
        plot_individual_trials!(lg3, "J";plot_joint=false, plottype=plottype, label=["F","G","H","I"],add_legend=false, shuffle_responses=false, shuffle_time=false, shuffle_trials=true, combine_subjects=true, save_all_β=true, shuffle_within_locations=true, t1=35.0, use_log=true)
        Label(lg3[1,0], "Monkey J", tellwidth=true, tellheight=false, rotation=π/2)
        lg31 = GridLayout(lg3[1,3])
        axj = plot_joint_regression!(lg31,"J";show_null_distr=false)
        llj = axj.finallimits[]
        llw = axw.finallimits[]
        ymin = min(llj.origin[2], llw.origin[2])
        ymax = ymin + max(llj.widths[2], llw.widths[2])
        ylims!(axj, ymin, ymax)
        ylims!(axw, ymin, ymax)
        # TODO: This doesn't seem to work
        linkyaxes!(axj, axw)
        rowsize!(fig.layout, 1, Relative(0.2))
        fig
    end
end

"""
Show a scatter plot of reaction time vs  each of the variable in `qdata` for one set of trials
"""
function plot_individual_trials(qdata,area::String;kvs...)
    trialidx = qdata["trialidx"][:,1]
    subject = qdata["subjects"][1]

end

function plot_individual_trials(subject::String, ;kvs...)
    with_theme(PlotUtils.plot_theme) do
        fig = Figure(size=(700,300))
        lg = GridLayout(fig[1,1])
        plot_individual_trials!(lg, subject;kvs...)
        fig
    end
end

function plot_individual_trials!(lg, subject::String, ;npoints=100, show_dlpfc=false, plottype=:boxplot, label::Union{Nothing, Vector{String}}=nothing, add_legend=true, xlabelvisible=true, t0=0.0, speedvar::Symbol=:SM, plot_joint=true, kvs...)
    qdataz = compute_regression(;subjects=[subject], nruns=100, sessions=:all, varnames=[:Z,:ncells,:xpos,:ypos], kvs...)
    qdatal = compute_regression(;subjects=[subject], nruns=100, sessions=:all, varnames=[:L,:ncells,:xpos,:ypos], kvs...)
    qdatass = compute_regression(;subjects=[subject], nruns=100, sessions=:all, varnames=[speedvar,:ncells,:xpos,:ypos], kvs...)

    qdatalz = compute_regression(;subjects=[subject], nruns=100, sessions=:all, varnames=[:L, :Z,:ncells,:xpos,:ypos], kvs...)
    qdatazss = compute_regression(;subjects=[subject], nruns=100, sessions=:all, varnames=[speedvar, :Z, :ncells,:xpos,:ypos], kvs...)
    qdataplss = compute_regression(;subjects=[subject], nruns=100, sessions=:all, varnames=[:L, speedvar, :ncells,:xpos,:ypos], kvs...)

    qdatazplss = compute_regression(;subjects=[subject], nruns=100, sessions=:all, varnames=[:L, speedvar, :Z, :ncells,:xpos,:ypos], kvs...)
    bins = qdatazss["bins"]
    bidx = searchsortedfirst(bins, t0)

    r² = Dict()
    r²z = Dict()
    areas = ["fef", "dlpfc"]
    for area in areas 
        r²[area] = Dict()
        r²z[area] = Dict()
        _r² = r²[area]
        _r²z = r²z[area]
        _r²["MP"] = qdataz[area]["r²"][bidx,:]
        _r²z["MP"] = qdataz[area]["r²_shuffled"][bidx,:]
        _r²["PL"] = qdatal[area]["r²"][bidx,:]
        _r²z["PL"] = qdatal[area]["r²_shuffled"][bidx,:]
        _r²["AS"] = qdatass[area]["r²"][bidx,:]
        _r²z["AS"] = qdatass[area]["r²_shuffled"][bidx,:]
        _r²["MP+PL"] = qdatalz[area]["r²"][bidx,:]
        _r²z["MP+PL"] = qdatalz[area]["r²_shuffled"][bidx,:]
        _r²["MP+AS"] = qdatazss[area]["r²"][bidx,:]
        _r²z["MP+AS"] = qdatazss[area]["r²_shuffled"][bidx,:]
        _r²["PL+AS"] = qdataplss[area]["r²"][bidx,:]
        _r²z["PL+AS"] = qdataplss[area]["r²_shuffled"][bidx,:]
        _r²["MP+PL+AS"] = qdatazplss[area]["r²"][bidx,:]
        _r²z["MP+PL+AS"] = qdatazplss[area]["r²_shuffled"][bidx,:]

    end
    @debug "Synerogy" mean((r²["fef"]["PL+AS"] - (r²["fef"]["PL"] + r²["fef"]["AS"]))./r²["fef"]["PL+AS"])
    acolor = [PlotUtils.fef_color, PlotUtils.dlpfc_color]
    
    lg2 = GridLayout(lg[1,1])
    β = fill(0.0, 3)
    for (ii,qdata) in enumerate([qdataz,qdatal,qdatass])
        β[ii] = qdata["fef"]["β"][1,bidx,1]
    end
    if label != nothing
        plotlabel = label[1:end-1]
    else
        plotlabel = nothing
    end
    axes = plot_individual_trials!(lg2, qdatal["trialidx"][:,1],"fef", subject, [:Z,:L,:SM];β=β, npoints=npoints, xlabelvisible=xlabelvisible, plotlabel=plotlabel, kvs...)
    for (k,ax) in zip(["MP","PL","AS"], axes)
        μ = mean(r²["fef"][k])
        ax.title = @sprintf("r² = %0.2f", μ)
        ax.titlefont = :regular
    end
    ax = Axis(lg[1,2])
    offset = 1

    if plot_joint
        combis = ["MP","PL","MP+PL","AS","MP+AS","PL+AS","MP+PL+AS"]
    else
        combis = ["MP","PL","AS"]
    end

    for k in combis
        for (cc,a) in zip(acolor,areas) 
            μ = mean(r²[a][k])
            zscore = (μ - mean(r²z[a][k]))/std(r²z[a][k])
            if zscore > 3.72 
                aa = "**"
            elseif zscore > 2.33 
                aa = "*"
            else
                aa = ""
            end
            if a == "dlpfc" && show_dlpfc == false
                # just print the value
                print("DLPFC: \n")
                print("\t$k: zscore = $(zscore)\n")
            else
                print("FEF: \n")
                print("\t$k: zscore = $(zscore)\n")
                if plottype == :boxplot
                    boxplot!(ax, fill(offset,100), r²[a][k],color=cc, markersize=5px)
                else
                    barplot!(ax, [offset], [mean(r²[a][k])], color=cc)
                end
                text!(ax, offset, μ, text=aa, align=(:center, :bottom))
                offset += 1
            end
        end
    end
    ax.xticks = (range(1.0, step=1, length=length(combis)),combis)
    ax.xticklabelrotation = -π/3
    ax.ylabel = "r²"
    if add_legend
        Legend(lg[1,2], [PolyElement(color=c) for c in acolor], ["FEF","DLPFC"], valign=:top,
                halign=:right, tellwidth=false, margin=(10.0, -15.0, 0.0, -15.0), labelsize=12)
    end
    if label !== nothing
        #Label(lg[1,1,TopLeft()], label[1])
        Label(lg[1,2,TopLeft()], label[end])
    end
    colsize!(lg,1,Relative(0.7))

    # plot comparison fef vs dlpfc for individual as well as combinations
    #trialidx = qdata["trialidx"][:,1]

end

function plot_individual_trials(trialidx::Vector{Int64},area::String,subject::String, varnames::Vector{Symbol};kvs...)
    with_theme(PlotUtils.plot_theme) do
        fig = Figure()
        lg = Gridlayout(fig[1,1])
        plot_individual_trials!(lg, trialidx, area, subject, varnames;kvs...)
        fig
    end
end

"""
Return variables from `allvars` use `varnames`
"""
function filter_regression_variables(allvars::NamedTuple,varnames::Vector{Symbol})
    vars = Any[]
    use_varnames = Symbol[]
    for vv in varnames
        xv = allvars[v]
        if eltype(xv) <: Integer && length(unique(xv)) == 1
                continue
        else
            push!(vars, xv)
            push!(use_varnames, vv)
        end
    end
    vars, use_varnames
end

"""
Test whether the regression model containing MP and PL fits significantly better than one without MP at the onset of the 
go-cue period for Monkey W, using the F-test.
"""
function test_hiearchical_dependence()
    ppsth,tlabels,_trialidx, rtimes = load_data(nothing;area="fef", raw=true)
    Z,L,EE, MM, SS, FL, lrt,label,ncells,bins,sessionidx = get_regression_data(ppsth,tlabels,_trialidx, rtimes, "W"; mean_subtract=true, variance_stabilize=true,window=50.0,use_log=true)
    bidx = searchsortedfirst(bins, 0.0)
    xpos = [p[1] for p in Utils.location_position["W"]][label]
    ypos = [p[2] for p in Utils.location_position["W"]][label]
    β1, Δβ, pv, r², rss1, varidx = compute_regression(lrt, L[:,bidx], ncells,xpos,ypos;exclude_pairs=[(3,4)],save_all_β=true)
    β2, Δβ, pv, r², rss2, varidx = compute_regression(lrt, L[:,bidx], Z[:,bidx], ncells,xpos,ypos;exclude_pairs=[(4,5)],save_all_β=true)
    p1 = length(β1)
    p2 = length(β2)
    n = size(L,1)
    F = (rss1 - rss2)/(p2-p1)
    F /= rss2/(n-p2)
    1.0 - cdf(FDist(p2-p1, n-p2), F[1])
end

function plot_individual_trials!(lg, trialidx::Vector{Int64},area::String,subject::String, varnames::Vector{Symbol};plotlabel::Union{Vector{String},Nothing}=nothing, β::Union{Nothing, Vector{Float64}}=nothing, npoints=length(trialidx), xlabelvisible=true, recording_side::Utils.RecordingSide=Utils.BothSides(), t0=0.0, sessions=:all, kvs...)

    vnames = Dict(:Z=>"MP", :L => "PL", :SM=>"AS")
    ppsth,tlabels,_trialidx, rtimes = load_data(nothing;area=area,raw=true, kvs...)
    Z,L,EE, MM, SS, FL, SM, lrt,label,ncells,bins,sessionidx = get_regression_data(ppsth,tlabels,_trialidx, rtimes, subject; mean_subtract=true, variance_stabilize=true,window=50.0, kvs...);
    bidx = searchsortedfirst(bins, t0)
    if subject == "J"
        tidx = findall(label.!=9)
    else
        tidx = 1:size(Z,1)
    end
    if sessions != :all
        tidx = tidx[findall(in(sessions), sessionidx[tidx])]
    end
    # use only trials associated with targets in the specfied hemifield
    tidx = tidx[findall(in(Utils.get_recording_side(recording_side, subject)), label[tidx])]

    xpos = [p[1] for p in Utils.location_position[subject]][label[tidx]]
    ypos = [p[2] for p in Utils.location_position[subject][label[tidx]]]
    allvars = (Z=Z[tidx,bidx:bidx], L=L[tidx,bidx:bidx],EE=EE[tidx,bidx:bidx],MM=MM[tidx,bidx:bidx],
               SS=SS[tidx,bidx:bidx],FL=FL[tidx,bidx:bidx], SM=SM[tidx,bidx:bidx], ncells=ncells[tidx], xpos=xpos, ypos=ypos)
    # exclude ncells if we are only doing one session
    #vars = [lrt[tidx], L[tidx,:], Z[tidx,:], EE[tidx,:], MM[tidx,:], xpos, ypos] 
    vars = Any[lrt[tidx]]
    use_varnames = Symbol[]
    for vv in varnames
        if vv == :ncells 
            if length(unique(ncells)) > 1
                push!(vars, ncells)
                push!(use_varnames, :ncells)
            end
        else
            push!(vars, allvars[vv])
            push!(use_varnames, vv)
        end
    end
    if all(in(use_varnames).([:xpos, :ypos]))
        exclude_pairs = [(findfirst(use_varnames.==:xpos),findfirst(use_varnames.==:ypos))]
    else
        exclude_pairs = Tuple{Int64, Int64}[]
    end
    β0 = fill(0.0, 2, length(varnames))
    # joint
    if β === nothing
        β,_,_,_,_ = compute_regression(repeat(trialidx,1,1),vars...;exclude_pairs=exclude_pairs,save_all_β=true)
    end
    #individual
    for ii in 1:length(varnames)
       # _β,_,_,_,_ = compute_regression(repeat(trialidx,1,1),vars[[1,ii+1]]...;exclude_pairs=exclude_pairs,save_all_β=true)
       # β0[:,ii] = _β
        β0[2,ii] = mean(vars[1] .- β[ii].*vars[ii+1])
    end
    # now plot
    la = lowercase(area)
    if la == "dlpfc"
        acolor = PlotUtils.dlpfc_color
    elseif la == "fef"
        acolor = PlotUtils.fef_color
    else
        acolor = Makie.wong_colors()[1]
    end
    axes = [Axis(lg[1,i],xticks=WilkinsonTicks(3)) for i in 1:length(varnames)]
    linkyaxes!(axes...)
    for (ii,ax) in enumerate(axes)
        _tidx = sort(shuffle(1:length(trialidx))[1:min(npoints,length(trialidx))])
        scatter!(ax,  vars[1+ii][trialidx[_tidx],1], vars[1][trialidx[_tidx]],color=acolor,markersize=5px)
        ax.xlabel = vnames[varnames[ii]]
        ax.xlabelvisible = xlabelvisible
        # this is hard to interpret without the context of the other variables, so
        # we probably don't want to show them
        ablines!(ax, β0[end,ii], β[ii],color=:black)
    end
    for ax in axes[2:end]
        ax.yticklabelsvisible = false
    end
    axes[1].ylabel = "log(rt)"
    if plotlabel !== nothing
        for (i,l) in enumerate(plotlabel)
            Label(lg[1,i,TopLeft()], l)
        end
    end
    axes
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

"""

"""
function plot_regression2(rt, L, βl, βx, βy,x,y)
    pos = unique(zip(x,y))
    nt = length(rt)
    tcolors = fill(Makie.wong_colors()[1],nt)
    a = fill(0.0, length(pos))
    b = fill(0.0, length(pos))
    for (ii,_pos) in enumerate(pos)
        tidx = findall((x.==_pos[1]).&(y.==_pos[2]))
        tcolors[tidx] .= Makie.wong_colors()[ii]
        a[ii] = βl + βx*_pos[1] + βy*_pos[2]
        b[ii] = mean(rt - a[ii]*L)
    end
    with_theme(PlotUtils.plot_theme) do
        fig = Figure()
        ax = Axis(fig[1,1])
        scatter!(ax, L, rt,color=tcolors)
        for ii in 1:length(a)
            ablines!(ax, b[ii], a[ii],color=Makie.wong_colors()[ii])
        end
        fig
    end
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
    dm,idx1 = Trajectories.compute_triangular_path_length(traj1)
    dm,idx2 = Trajectories.compute_triangular_path_length(traj2)
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
        PlotUtils.plot_normals!(ax, traj1;ss=(traj,i)->PlotUtils.advance(traj,i,delta), linelength=0.16,color="red")
        lines!(ax, traj2[:,1], traj2[:,2],color="blue")
        linesegments!(ax, traj2[tidx2,1], traj2[tidx2,2],color="blue",linestyle=:dot)
        PlotUtils.plot_normals!(ax, traj2;ss=(traj,i)->PlotUtils.advance(traj,i,delta), linelength=0.16,color="blue")
    
        lines!(ax2, traj2[:,1], traj2[:,2],color="blue", label="Short RT")
        PlotUtils.plot_normals!(ax2, traj2;ss=(traj,i)->PlotUtils.advance(traj,i,delta), linelength=0.16,color="blue")
    
        lines!(ax2, traj3[:,1], traj3[:,2],color="red", label="Long RT")
        PlotUtils.plot_normals!(ax2, traj3;ss=(traj,i)->PlotUtils.advance(traj,i,delta/2.0), linelength=0.16,color="red")
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

"""
Set up variable for regression
"""
function get_regression_variables(vars...;bidx=1, trialidx=1:size(vars[1],1), kvs...)
    _tidx = 1:length(trialidx)
    do_skip = false
    for Z in vars 
        _tidx = intersect(_tidx,findall(isfinite, Z[trialidx,bidx]))
        if isempty(_tidx)
            do_skip = true
            break
        end
    end
    if do_skip
        return nothing
    end
    trialidx = trialidx[_tidx]
    # include all but the first variable
    X = hcat([ndims(Z)== 2 ? Z[trialidx,bidx] : Z[trialidx] for Z in vars]...)
    X, trialidx
end

function compute_regression_with_residuals(subject::String, varnames::Vector{Symbol}, area="fef";t0=0.0, nruns=100, kvs...)
    ppsth,tlabels,_trialidx, rtimes = load_data(nothing;area="fef", raw=true)
    Z,L,EE, MM, SS, FL, SM, lrt,label,ncells,bins,sessionidx = get_regression_data(ppsth, tlabels, _trialidx, rtimes, subject;tmin=t0, tmax=t0, kvs...)
    
    if subject == "J"
        tidx = findall(label.!=9)
    else
        tidx = 1:size(Z,1)
    end
    xpos = [p[1] for p in Utils.location_position[subject][label[tidx]]]
    ypos = [p[2] for p in Utils.location_position[subject][label[tidx]]]

    allvars = (Z=Z[tidx,:], L=L[tidx,:],EE=EE[tidx,:],MM=MM[tidx,:],
            SS=SS[tidx,:],FL=FL[tidx,:], SM=SM[tidx,:], ncells=ncells[tidx], xpos=xpos, ypos=ypos)

    use_varnames = Symbol[]

    vars = Any[]
    for kk in varnames
        vv = allvars[kk]
        if length(unique(vv)) > 1
            push!(vars, vv)
            push!(use_varnames, kk)
        end
    end
    if all(in(use_varnames[2:end]).([:xpos, :ypos]))
        exclude_pairs = [(findfirst(use_varnames[2:end].==:xpos),findfirst(use_varnames[2:end].==:ypos))]
    else
        exclude_pairs = Tuple{Int64, Int64}[]
    end

    # full model
    X1, trialidx1 = get_regression_variables(vars[2:end]...;exclude_pairs=exclude_pairs)
    X0, trialidx0 = get_regression_variables(vars[1])
    trialidx = intersect(trialidx1, trialidx0)
    tidx1 = findall(in(trialidx), trialidx1)
    tidx0 = findall(in(trialidx), trialidx0)

    @show size(X1) size(lrt[trialidx])
    lreg1 = LinearRegressionUtils.llsq_stats(X1[tidx1,:], lrt[trialidx];do_interactions=true,exclude_pairs=exclude_pairs)

    # residual
    lreg0 = LinearRegressionUtils.llsq_stats(X0[tidx0,:], lreg1.residual[tidx1];do_interactions=false)

    r² = lreg0.r²
    r²s = fill(0.0, nruns)
    @showprogress "Shuffling..." for i in 1:nruns
        Z,L,EE, MM, SS, FL, SM, lrt,label,ncells,bins,sessionidx = get_regression_data(ppsth, tlabels, _trialidx, rtimes, subject;do_shuffle_trials=true, shuffle_within_location=true, tmin=t0,tmax=t0, kvs...)
        allvars = (Z=Z[tidx,:], L=L[tidx,:],EE=EE[tidx,:],MM=MM[tidx,:],
               SS=SS[tidx,:],FL=FL[tidx,:], SM=SM[tidx,:], ncells=ncells[tidx], xpos=xpos, ypos=ypos)

        X0, trialidx0 = get_regression_variables(allvars[use_varnames[1]])
        trialidx = intersect(trialidx1, trialidx0)
        tidx0 = findall(in(trialidx), trialidx0)
        lreg0 = LinearRegressionUtils.llsq_stats(X0[tidx0,:], lreg1.residual[tidx1];do_interactions=false)
        r²s[i] = lreg0.r²
    end
    r²,r²s
end

function plot_joint_regression(;kvs...)
    with_theme(plot_theme) do
        fig = Figure(size=(400,300))
        lg1 = GridLayout(fig[1,1])
        plot_joint_regression!(lg1, "W";kvs...)
        Label(lg1[1,0], "Monkey W", rotation=π/2, tellheight=false)
        lg2 = GridLayout(fig[2,1])
        plot_joint_regression!(lg2, "J";kvs...)
        Label(lg2[1,0], "Monkey J", rotation=π/2, tellheight=false)
        fig
    end
end 

function plot_joint_regression!(lg, subject::String;redo=false, show_null_distr=true)
    fname = "joint_regression_subjec_$(subject).jld2"
    if !redo && isfile(fname)
        r², r²s = JLD2.load(fname, "r²", "r²s") 
    else
        r² = Dict()
        r²s = Dict()
        r²["AS→MP"], r²s["AS→MP"] = compute_regression_with_residuals(subject, [:SM, :Z,:ncells,:xpos,:ypos];t0=0.0, nuns=100,  use_midpoint=false,align=:cue, tt=65.0, shuffle_responses=false,shuffle_time=false,shuffle_trials=true,combine_subjects=true,save_all_β=true,shuffle_within_locations=true,t1=35.0, use_log=true, use_new_energy_point_algo=false)
        r²["PL→MP"], r²s["PL→MP"] = compute_regression_with_residuals(subject, [:L, :Z,:ncells,:xpos,:ypos];t0=0.0, nuns=100,  use_midpoint=false,align=:cue, tt=65.0, shuffle_responses=false,shuffle_time=false,shuffle_trials=true,combine_subjects=true,save_all_β=true,shuffle_within_locations=true,t1=35.0, use_log=true, use_new_energy_point_algo=false)
        r²["AS→PL"], r²s["AS→PL"] = compute_regression_with_residuals(subject, [:SM, :L,:ncells,:xpos,:ypos];t0=0.0, nuns=100,  use_midpoint=false,align=:cue, tt=65.0, shuffle_responses=false,shuffle_time=false,shuffle_trials=true,combine_subjects=true,save_all_β=true,shuffle_within_locations=true,t1=35.0, use_log=true, use_new_energy_point_algo=false)
        r²["PL→AS"], r²s["PL→AS"] = compute_regression_with_residuals(subject, [:L, :SM,:ncells,:xpos,:ypos];t0=0.0, nuns=100,  use_midpoint=false,align=:cue, tt=65.0, shuffle_responses=false,shuffle_time=false,shuffle_trials=true,combine_subjects=true,save_all_β=true,shuffle_within_locations=true,t1=35.0, use_log=true, use_new_energy_point_algo=false)
        r²["MP→PL"], r²s["MP→PL"] = compute_regression_with_residuals(subject, [:Z, :L,:ncells,:xpos,:ypos];t0=0.0, nuns=100,  use_midpoint=false,align=:cue, tt=65.0, shuffle_responses=false,shuffle_time=false,shuffle_trials=true,combine_subjects=true,save_all_β=true,shuffle_within_locations=true,t1=35.0, use_log=true, use_new_energy_point_algo=false)
        r²["MP→AS"], r²s["MP→AS"] = compute_regression_with_residuals(subject, [:Z, :SM,:ncells,:xpos,:ypos];t0=0.0, nuns=100,  use_midpoint=false,align=:cue, tt=65.0, shuffle_responses=false,shuffle_time=false,shuffle_trials=true,combine_subjects=true,save_all_β=true,shuffle_within_locations=true,t1=35.0, use_log=true, use_new_energy_point_algo=false)
        JLD2.save(fname, Dict("r²"=>r²,"r²s"=>r²s ))
    end
    for k in keys(r²s) 
        B = fit(Beta, r²s[k])
        r²s[k] = Dict("percentiles" => percentile(r²s[k], [5,50,95]), "distr"=>B)
    end

    ax = Axis(lg[1,1])
    _keys = collect(keys(r²))
    _keys = ["MP→PL","MP→AS","PL→AS","AS→PL"]
    barplot!(ax, 1:length(_keys), [r²[k] for k in _keys], color=PlotUtils.fef_color)
    ymax = maximum([r²[k] for k in _keys])
    if show_null_distr
        scatter!(ax, 1:length(_keys), [r²s[k]["percentiles"][2] for k in _keys],color=:black)
        rangebars!(ax, 1:length(_keys), [r²s[k]["percentiles"][1] for k in _keys],[r²s[k]["percentiles"][3] for k in keys(r²)],color=:black)
    else
        # just indicate significance
        for (ii,_k) in enumerate(_keys)
            y = r²[_k]
            B = r²s[_k]["distr"]
            pv = 1 - cdf(B, y)
            @show pv
            tt = ""
            if pv < 0.001
                tt = "**"
            elseif pv < 0.01
                tt = "*"
            end
            text!(ax, ii, y,text=tt, align=(:center,:bottom))
            ymax = 1.05*ymax
        end
    end
    ax.xticks = (1:length(_keys), _keys)
    ax.xticklabelrotation = -π/3
    ax.ylabel = "residual r²"
    ylims!(ax, 0.0, ymax)
    ax
end
end #module