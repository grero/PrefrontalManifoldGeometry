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

include("utils.jl")
include("trajectories.jl")
include("plot_utils.jl")

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
function get_regression_data(ppsth,tlabels,trialidx,rtimes,subject::Union{Nothing,String};rtmin=120.0, rtmax=300.0, window=35.0, Δt=15.0,t1=35.0, realign=true, do_shuffle=false, do_shuffle_responses=false, do_shuffle_time=false, do_shuffle_trials=false, shuffle_within_locations=false, nruns=100,smooth_window::Union{Nothing, Float64}=nothing, use_midpoint=false,tt=65.0, use_log=false, kvs...)

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
    Z = fill(0.0,n_tot_trials, size(ppsth.counts,1))
    L = fill!(similar(Z),NaN) 
    EE = fill!(similar(Z),NaN)
    MM = fill!(similar(Z),NaN) # modified enerngy
    SS = fill!(similar(Z), NaN) # speed
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
        _lrt = log.(_rtime)
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
        idxp = searchsortedfirst(bins, tt-Δt-window)
        MM[offset+1:offset + _nt,idxp], V = get_projected_energy(X, map(x->x .+ (idxt-1), qidxs),_label)
        for j in 1:size(X,1)
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
                Z[offset+1:offset+_nt,j]  .= z
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
                if use_midpoint
                    # midpoint
                    ip = div(length(_idxv),2)+1
                else
                    # point of high energy
                    ee = dropdims(sum(abs2, Xs,dims=2),dims=2)
                    ip = argmax(ee)
                end
                if smooth_window !== nothing
                    Xs = mapslices(x->Utils.gaussian_smooth(x, bins[_idxv], smooth_window), Xs, dims=1)
                end
                L[offset+j2,j] = compute_triangular_path_length(Xs, ip)
                # energy
                EE[offset+j2,j] = maximum(sum(abs2, Xs .- Xs[1:1,:],dims=2))
                # modified energy; projected onto the line
                MM[offset+j2,j] = get_projected_energy(Xs, V[:,l])
                # avg speed
                SS[offset+j2,j] = mean(sqrt.(sum(abs2,diff(Xs,dims=1),dims=2)))

            end
        end
        append!(rt, _lrt)
        append!(label, _label)
        append!(ncells, fill(size(X,3), length(_label)))
        append!(sessionidx, fill(ii, length(_label)))
        offset += _nt
    end
    if use_log
        rt = log.(rt)
    end
    Z[1:offset,:], L[1:offset,:], EE[1:offset,:,], MM[1:offset,:], SS[1:offset,:], rt, label, ncells, bins, sessionidx
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
        β[:,:,i],Δβ[:,:,i],pv[:,i],r²[:,i],_ = compute_regression(args...;trialidx=trialidx[:,i],kvs...)
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
    if shuffle_trials
        sidx = shuffle(trialidx)
    else
        sidx = trialidx
    end
    varidx = nothing
    for i in axes(L,2) 
        do_skip = false
        for Z in [L,args...]
            if ndims(Z)==2
                if !all(isfinite.(Z[trialidx,i]))
                    do_skip = true
                    break
                end
            end
        end
        if do_skip
            continue
        end 
        # run two regression models; one without path length L and one with
        #X_no_L = [Z[trialidx,i] xpos[trialidx] ypos[trialidx] ncells[trialidx] EE[trialidx,i]]
        X_no_L = hcat([ndims(Z)== 2 ? Z[trialidx,i] : Z[trialidx] for Z in args]...)
        #X_with_L = [Z[trialidx,i] xpos[trialidx] ypos[trialidx] L[trialidx,i] ncells[trialidx] EE[trialidx,i]]
        X_with_L = [L[trialidx,i] X_no_L]

        #lreg_no_L = LinearRegressionUtils.llsq_stats(X_no_L, rt[sidx];do_interactions=true, exclude_pairs=[(2,3)])
        #lreg_with_L = LinearRegressionUtils.llsq_stats(X_with_L, rt[sidx];do_interactions=true, exclude_pairs=[(2,3)])
        # subtract one from the exclude pair index
        _exclude_pairs = [(i-1,j-1) for (i,j) in exclude_pairs]
        lreg_no_L = LinearRegressionUtils.llsq_stats(X_no_L, rt[sidx];do_interactions=true,exclude_pairs=_exclude_pairs, kvs...)
        lreg_with_L = LinearRegressionUtils.llsq_stats(X_with_L, rt[sidx];do_interactions=true,exclude_pairs=exclude_pairs, kvs...)

        # compute the F-stat for whether adding the path length results in a significantly better fit
        n = length(rt)
        rss2 = lreg_with_L.rss
        p2 = length(lreg_with_L.β)
        rss1 = lreg_no_L.rss
        p1 = length(lreg_no_L.β)
        F = (rss1 - rss2)/(p2-p1)
        F /= rss2/(n-p2)
        pv[i] = 1.0 - cdf(FDist(p2-p1, n-p2), F)
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
    end
    β,Δβ, pv, r², varidx
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

function compute_regression(;redo=false, varnames=[:L, :Z, :ncells, :xpos, :ypos], subjects=["J","W"], sessions::Union{Vector{Int64},Symbol}=:all, combine_subjects=false, tt=65.0,nruns=100, use_midpoint=false, shuffle_responses=false,shuffle_time=false, shuffle_trials=false, check_only=false, save_all_β=false, kvs...)
    # TODO: Add option for combining regression for both animals
    q = UInt32(0)
    if subjects != ["J","W"]
        q = crc32c(string((:subjects=>subjects)),q)
    end
    if sessions != :all
        q = crc32c(string((:sessions=>sessions)),q)
    end
    q = crc32c(string(:varnames=>varnames),q)
    q = crc32c(string((:use_midpoint=>use_midpoint)),q)
    if shuffle_responses
        q = crc32c(string((:shuffle_responses=>true)),q)
    end
    if shuffle_time
        q = crc32c(string((:shuffle_time=>true)),q)
    end
    if shuffle_trials
        q = crc32c(string((:shuffle_trials=>true)),q)
    end
    if combine_subjects
        q = crc32c(string((:combine_subjects=>true)),q)
    end
    if nruns != 100
        q = crc32c(string((:nruns=>nruns)),q)
    end
    if save_all_β
        q = crc32c(string((:save_all_β=>true)),q)
    end
    for k in kvs
        q = crc32c(string(k),q)
    end
    qs = string(q, base=16)
    fname = "path_length_regression_$(qs).jld2"
    if check_only
        return isfile(fname)
    end
    if isfile(fname) && redo == false
        qdata = JLD2.load(fname)
    else
        qdata = Dict()
        bins = Float64[]
        for area in ["fef","dlpfc"]
            qdata[area] = Dict()
            Za = Vector{Matrix{Float64}}(undef, length(subjects))
            La = Vector{Matrix{Float64}}(undef, length(subjects))
            EEa = Vector{Matrix{Float64}}(undef, length(subjects))
            MMa = Vector{Matrix{Float64}}(undef, length(subjects))
            ncellsa = Vector{Vector{Int64}}(undef, length(subjects))
            xposa = Vector{Vector{Float64}}(undef, length(subjects))
            yposa = Vector{Vector{Float64}}(undef, length(subjects))
            lrta = Vector{Vector{Float64}}(undef, length(subjects))
            sessionidxa = Vector{Vector{Int64}}(undef, length(subjects))

            ppsth,tlabels,trialidx, rtimes = load_data(nothing;area=area,raw=true, kvs...)

            for (ss, subject) in enumerate(subjects)
                # load the data here so we don't have to do it more than once
                # TODO: Do the shuffling here (hic misce)
                Z,L,EE, MM, lrt,label,ncells,bins,sessionidx = get_regression_data(ppsth,tlabels,trialidx, rtimes, subject; mean_subtract=true, variance_stabilize=true,window=50.0,use_midpoint=use_midpoint, kvs...);
                if subject == "J"
                    tidx = findall(label.!=9)
                else
                    tidx = 1:size(Z,1)
                end
                if sessions != :all
                    tidx = tidx[findall(in(sessions), sessionidx[tidx])]
                end
                xpos = [p[1] for p in Utils.location_position[subject]][label[tidx]]
                ypos = [p[2] for p in Utils.location_position[subject][label[tidx]]]
                Za[ss] = Z[tidx,:]
                La[ss] = L[tidx,:]
                EEa[ss] = EE[tidx,:]
                MMa[ss] = MM[tidx,:]
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
            EE = cat(EEa..., dims=1)
            MM = cat(MMa..., dims=1)
            ncells = cat(ncellsa..., dims=1)
            xpos = cat(xposa..., dims=1)
            ypos = cat(yposa..., dims=1)
            lrt = cat(lrta..., dims=1)
            allvars = (Z=Z, L=L,EE=EE,MM=MM,ncells=ncells, xpos=xpos, ypos=ypos)
            # exclude ncells if we are only doing one session
            #vars = [lrt[tidx], L[tidx,:], Z[tidx,:], EE[tidx,:], MM[tidx,:], xpos, ypos] 
            vars = Any[lrt]
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
                    _Z,_L,_EE, _MM, _lrt,_label,_ncells,bins,_sessionidx = get_regression_data(ppsth,tlabels,trialidx,rtimes,subject;mean_subtract=true, variance_stabilize=true,window=50.0,use_midpoint=use_midpoint, do_shuffle=do_shuffle, do_shuffle_responses=do_shuffle_responses,do_shuffle_time=do_shuffle_time,do_shuffle_trials=do_shuffle_trials,kvs...);
                    if subject == "J"
                        tidx = findall(_label.!=9)
                    else
                        tidx = 1:size(_Z,1)
                    end
                    if sessions != :all
                        tidx = tidx[findall(in(sessions), _sessionidx[tidx])]
                    end
                    _xpos = [p[1] for p in Utils.location_position[subject]][_label[tidx]]
                    _ypos = [p[2] for p in Utils.location_position[subject][_label[tidx]]]
                    Za[ss] = _Z[tidx,:]
                    La[ss] = _L[tidx,:]
                    EEa[ss] = _EE[tidx,:]
                    MMa[ss] = _MM[tidx,:]
                    ncellsa[ss] = _ncells[tidx]
                    xposa[ss] = _xpos
                    yposa[ss] = _ypos
                    lrta[ss] = _lrt[tidx]
                end
                # need to again concatente
                Z = cat(Za...,dims=1)
                L = cat(La...,dims=1)
                EE = cat(EEa..., dims=1)
                MM = cat(MMa..., dims=1)
                ncells = cat(ncellsa..., dims=1)
                xpos = cat(xposa..., dims=1)
                ypos = cat(yposa..., dims=1)
                lrt = cat(lrta..., dims=1)
                allvars = (Z=Z, L=L,EE=EE,MM=MM,ncells=ncells, xpos=xpos, ypos=ypos)

                vars = Any[lrt]
                for vv in use_varnames
                    push!(vars, allvars[vv])
                end
                _β,_,_,_r²,_ = compute_regression(vars...;exclude_pairs=exclude_pairs,shuffle_trials=false,save_all_β=save_all_β)
                β_S[:,:,r] .= _β
                r²_S[:,r] .= _r²
            end
            qdata[area]["β_shuffle"]=β_S
            qdata[area]["r²_shuffled"]=r²_S
        end
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

function plot_β_comparison(qdata::Vector{T}, area::String, βindex::Int64;show_zscore=false, tt=65.0) where T <: Dict
    with_theme(PlotUtils.plot_theme) do
        fig = Figure(size=(500,400))
        ax = Axis(fig[1,1])
        vlines!(ax, [0.0, tt];color=[:black, :gray])
        for _qdata in qdata 
            bins = _qdata["bins"]
            _data = _qdata[area]
            bidx = findall(isfinite, dropdims(mean(_data["r²"],dims=2),dims=2))
            μ = dropdims(mean(_data["β"][βindex,bidx,:],dims=2),dims=2)
            if show_zscore
                βs = _data["β_shuffle"]
                μs = dropdims(mean(βs[βindex,bidx,:],dims=2),dims=2)
                σs = dropdims(std(βs[βindex,bidx,:],dims=2),dims=2)
                μ .= (μ - μs)./σs
            end
            lines!(ax, bins[bidx], μ,label=join(_qdata["subjects"]), linewidth=1.5)
        end
        if show_zscore
            ax.ylabel = "Z-scored β $(area)"
        else
            ax.ylabel="β"
        end
        ax.xlabel = "Time from go-cue [ms]"
        axislegend(ax,valign=:top, halign=:left)
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