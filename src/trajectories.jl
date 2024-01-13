using DataProcessingHierarchyTools
const DPHT = DataProcessingHierarchyTools
using StatsBase

"""
```julia
function compute_triangular_path_length(traj::Matrix{Float64})

Find the largest combined Euclidean distance from the first point to a point on the trajectory, and from that point to the last point.
```
"""
function compute_triangular_path_length(traj::Matrix{T},method=:normal, do_shuffle=false) where T <: Real
	nn = size(traj,1)
	dm = -Inf
	qidx = 0
	if method == :normal
		func = compute_triangular_path_length
	elseif  method == :sq
		func = compute_triangular_path_length_sq
	else
		error("Unknown method $method")
	end
	for i in 2:nn-1
		_d = func(traj, i)
		if _d > dm
			dm = _d
			qidx = i
		end
	end
	dm,qidx
end

"""
```
function compute_triangular_path_length2(traj::Matrix{Float64})

Return the path length of 3 points that best preserves the overall path length
```
"""
function compute_triangular_path_length2(traj::Matrix{T}) where T <: Real
	nn = size(traj,1)
	pl = sum(sqrt.(sum(abs2, diff(traj,dims=1),dims=2)))
	qidx = 0
	dm = Inf
	dq = 0.0
	for i in 2:nn-1
		dd = compute_triangular_path_length(traj, i)
		_dm = dd - pl
		_dm *= _dm
		if _dm < dm
			dm = _dm
			qidx = i
			dq = dd
		end
	end
	dq,qidx
end

function compute_triangular_path_length(traj::Matrix{T},i::Int64) where T <: Real 
	_d = sqrt(sum(abs2, traj[i,:]  - traj[1,:]))
	_d += sqrt(sum(abs2, traj[i,:] - traj[end,:]))
end

"""
```
function compute_triangular_path_length_sq(traj::Matrix{Float64},i::Int64)
````
Computes path length by summing up the square of the line-elements.
"""
function compute_triangular_path_length_sq(traj::Matrix{T},i::Int64) where T <: Real
	_d = sum(abs2, traj[i,:]  - traj[1,:])
	_d += sum(abs2, traj[i,:] - traj[end,:])
end

function compute_triangular_path_length(X::Array{T,3}, qidx::Matrix{Int64}, args...;do_shuffle=false) where T <: Real
	nn = qidx[2,:] - qidx[1,:] .+ 1
	path_lengths = fill(0.0, length(nn))
	for i in 1:length(path_lengths)
		_X = X[qidx[1,i]:qidx[2,i],i,:]
		if do_shuffle
			bidx = shuffle(1:nn[i])	
			_X = _X[bidx,:]
		end
		path_lengths[i],_ = compute_triangular_path_length(_X,args...)
	end
	path_lengths
end

function compute_triangular_path_length2(X::Array{T,3}, qidx::Matrix{Int64}) where T <: Real
	nn = qidx[2,:] - qidx[1,:] .+ 1
	path_lengths = fill(0.0, length(nn))
	for i in 1:length(path_lengths)
		path_lengths[i],_ = compute_triangular_path_length2(X[qidx[1,i]:qidx[2,i],i,:])
	end
	path_lengths
end

function get_path_length_and_rtime(subject::String, t0::Real, t1::Real, ;operation=:path_length, rtmin=120.0, rtmax=300.0, area="FEF", do_shuffle=false, kvs...)
	fname = joinpath("data", "ppsth_fef_cue_raw.jld2")
	ppstht, labelst, rtimest, trialidxt = JLD2.load(fname, "ppsth","labels", "rtimes", "trialidx")
	bins = ppstht.bins
	subject_index = findall(cell->DPHT.get_level_name("subject", cell)==subject, ppstht.cellnames)
    cellidx = get_area_index(ppstht.cellnames[subject_index], area)
    cellidx = subject_index[cellidx]

    all_sessions = DPHT.get_level_path.("session", ppstht.cellnames[cellidx])
    sessions = unique(all_sessions)
	lrt = Float64[]
	path_length = Float64[]
	ncells = fill(0, length(sessions))
	counts = Float64[]
	qridx = Int64[]
	countstats = Dict()
	offset = 0
	for (sessionidx,session) in enumerate(sessions)
		X, labels, rtimes = get_session_data(session, ppstht, trialidxt, labelst, rtimest, cellidx;rtime_min=rtmin, rtime_max=rtmax, mean_subtract=true, variance_stabilize=true)
		ncells[sessionidx] = size(X,3)
		# how many locations?
		nc = maximum(labels)
		ulabels = unique(labels)
		sort!(ulabels)
		nlabels = length(ulabels)
		for (ii,loc) in enumerate(locations[subject])
			Xl = X[:,labels.==loc, :]
			rtimel = rtimes[labels.==loc]
			qidx = get_transition_period(bins, rtimel, t0, t1)

			tridx = findall(qidx[2,:] - qidx[1,:] .+ 1 .>= 3)
			for jj in tridx
				push!(counts, mean(sum(abs2, Xl[qidx[1,jj]:qidx[2,jj],jj,:],dims=1)))
				qv = countmap(Xl[qidx[1,jj]:qidx[2,jj],jj,:][:])
				merge!(countstats, qv)
			end
			if operation == :path_length
				S = compute_triangular_path_length(Xl[:, tridx,:], qidx[:,tridx];do_shuffle=do_shuffle)
			elseif operation == :path_length2
				S = compute_triangular_path_length(Xl[:, tridx,:], qidx[:,tridx],:sq;do_shuffle=do_shuffle)
			elseif operation == :path_length_ref
				S = compute_ref_path_length(Xl[:,tridx,:],  qidx[:,tridx];do_shuffle=do_shuffle)
			elseif operation == :mean_speed
				S = fill(0.0, length(tridx))
				for jj in 1:length(S)
					trjj = tridx[jj]
					S[jj] = mean(sqrt.(sum(abs2, diff(Xl[qidx[1, trjj]:qidx[2,trjj],trjj,:],dims=1),dims=2)))
				end
			else
				error("Unkown operation $(operation). Only ``:path_length`` and ``mean_speed`` are currently recognised")
			end
			S ./= ncells[sessionidx]
			if any(isnan.(S))
				@warn "Nan encountered" sessionidx loc
			end
			# subtract the global mean. This probably means that points will shift upwards, since the previous mean,
			# which did not include the very short reaction time trials, would result in a higher value mean.  In others, now, we are 
			# subtracting a lower value, which shifts the points upwards.
			_lrt = log.(rtimel[tridx]) .- mean(log.(rtimel))
			append!(lrt,_lrt)
			append!(path_length, S)
			append!(qridx, offset .+ tridx)
			offset += size(Xl,2)
		end
	end
	path_length, lrt, qridx
end

"""
```julia
function get_transition_period(bins::AbstractVector{Float64}, rtime::AbstractVector{Float64}, t0::Float64, t1::Float64) where T <: Real
```
Get the start and end index of the transition period for each trial
"""
function get_transition_period(bins::AbstractVector{Float64}, rtime::AbstractVector{Float64}, t0::Float64, t1::Float64)
	ntrials = length(rtime)
	qidx = fill(0, 2, ntrials)
	idx0 = searchsortedfirst(bins, t0)
	qidx[1,:] .= idx0
	for i in 1:ntrials
		rt = rtime[i]
		idx1 = searchsortedlast(bins, rt-t1)
		qidx[2,i] = idx1
	end
	qidx
end
