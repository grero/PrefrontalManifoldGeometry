module Utils
using TestItems
using DataProcessingHierarchyTools
const DPHT = DataProcessingHierarchyTools 
using StatsBase
using Distributions

export get_session_data, location_position, ContraLatera, IpsiLateral

abstract type RecordingSide
end

struct IpsiLateral <: RecordingSide
end

struct ContraLateral <: RecordingSide
end

struct BothSides <: RecordingSide
end

struct Locations <: RecordingSide
	loc::Vector{Int64}
end

sessions_j = ["J/20140807/session01", "J/20140828/session01", "J/20140904/session01", "J/20140905/session01"]
sessions_w = ["W/20200106/session02", "W/20200108/session03", "W/20200109/session04", "W/20200113/session01", "W/20200115/session03", "W/20200117/session03", "W/20200120/session01", "W/20200121/session01"]

ncells = Dict("J" => [37, 29, 33, 25], "W" => [15, 18, 10, 13, 13, 16, 21, 21])
_ntrials = Dict("J" => [108, 140, 154, 196], "W" => [164, 93, 86, 227, 171, 144, 209, 177])

locations = Dict("J" => [1,2,3,4,6,7,8], "W" => [1:4;], "P" => [1,2,3,4,6,7,8,9])

location_mapping = Dict("W" => [2, 4, 1, 3],
				        "J" => [1,2,3,4,6,7,8,9],
	                    "P" => [1,2,3,4,6,7,8,9])

location_idx = Dict("W" => [3,9,1,7],
					"J" => [1,2,3,4,5,6,7,8,9],
					"P" => [1,2,3,4,5,6,7,8,9])

function get_recording_side(side::IpsiLateral, subject)
	if subject == "W" 
		return [1,3] 
	elseif subject == "J"
		[1,2,3]
	else
		error("Unkown subject $subject")
	end
end



function get_recording_side(side::ContraLateral, subject)
	if lowercase(subject) == "w"
		return [2,4]
	elseif lowercase(subject) == "j"
		return [7,8,9]
	else
		error("Unkown subject $subject")
	end
end

get_recording_side(side::BothSides, subject) = locations[subject]

get_recording_side(side::Locations, subject) = side.loc

@testitem "Recording side" begin
	location_position = PrefrontalManifoldGeometry.location_position
	idx = PrefrontalManifoldGeometry.Utils.get_recording_side(PrefrontalManifoldGeometry.ContraLateral(), "W")
	@test all([(x,y)->x > 0.0 for (x,y) in location_position["W"][idx]])

	idx = PrefrontalManifoldGeometry.Utils.get_recording_side(PrefrontalManifoldGeometry.ContraLateral(), "J")
	@test all([(x,y)->x > 0.0 for (x,y) in location_position["J"][idx]])
end

# physical location of each target on the screen, in (approximate) degress of visual angle
location_position = Dict("W" => [(-10.0, -10.0),(10.0, -10.0), (-10.0, 10.0), (10.0, 10.0)],
					     "J" => [(-10.0, 10.0),(-10.0, 0.0), (-10.0, -10.0),(0.0, 10.0),
						         (0.0, -10.0),(10.0, 10.0),(10.0, 0.0),(10.0, -10.0)],
						 "M" => [(0.0,0.0)] # model data
						)

areas = Dict("J" => Dict("array01" => "vFEF",
				"array02" => "dFEF",
				"array03" => "vDLPFC",
				"array04" => "dDLPFC",
				"array07" => "vDLPFC",
				"array08" => "dDLPFC"),
			"W" => Dict("array01" => "dDLPFC",
					"array02" => "dFEF",
					"array03" => "vDLPFC",
					"array04" => "vFEF"),
			"P" => Dict("array01" => "area8",
					    "array02" => "FEF",
						"array03" => "DLPFC",
						"array04" => "vDLPFC")
			)


function get_area_index(cells::Vector{String}, area::String)
	ncells= length(cells)
	qq = fill(false, ncells)
	for (i,c) in enumerate(cells)
		ss = DPHT.get_level_name("subject",c)
		aa = DPHT.get_level_name("array",c)
		if occursin(area,areas[ss][aa])
			qq[i] = true
		end
	end
	qq
end
			
"""
Convenience function to get labels corresponding to corner locations
"""
function filter_corner_locations(subject::String, label::Vector{Int64})
    if subject == "J" || subject == "P"
        corners = [1,3, 6,7]
        newlabel = [findfirst(corners.==l) for l in label]
    else
        newlabel = location_mapping[subject][label]
    end
    newlabel
end

function filter_contralateral_locations(subject::String, label::Vector{Int64})
    if subject == "J" || subject == "P"
		clocations = [6,7,8]
        newlabel = [findfirst(clocations.==l) for l in label]
	else
        newlabel = location_mapping[subject][label]
		newlabel = filter(in([3,4]), newlabel)
	end
	newlabel
end

function rebin2!(X2::AbstractMatrix{T}, X::AbstractMatrix{T}, bins::AbstractVector{Float64}, window::Float64;normalize=false) where T <: Real
	idx1 = 0
	j = 1
	while bins[j] + window <= bins[end]  
		idx0 = j
		idx1 = searchsortedlast(bins[1:end-1], bins[j] + window)
		X2[j:j,:] .= sum(X[idx0:idx1, :],dims=1)
		if normalize
			X2[j,:] ./= idx1-idx0+1
		end
		j += 1
	end
	X2, bins[1:j-1]
end

"""
Rebin the counts in `X` by summing over `window`.
"""
function rebin2(X::AbstractMatrix{T}, bins::AbstractVector{Float64}, window::Float64;kvs...) where T <: Real
	X2 = fill!(similar(X), 0.0)
	X2, bins2 = rebin2!(X2, X, bins, window;kvs...)	
	X2, bins2
end

function rebin2(X::Array{T,3}, bins::AbstractVector{Float64}, window::Float64;kvs...) where T <: Real
	X2 = fill!(similar(X), 0.0)
	rebin2!(X2, X, bins, window;kvs...)
end

function rebin2!(X2::Array{T,3}, X::Array{T,3}, bins::AbstractVector{Float64}, window::Float64;kvs...) where T <: Real
	_, bins2 = rebin2!(view(X2, :,:, 1), view(X, :,:, 1), bins, window;kvs...)
	for i in 2:size(X2, 3)
		rebin2!(view(X2, :,:, i), view(X, :,:, i), bins, window;kvs...)
	end
	X2,bins2
end

function get_session_data(session::String, ppsth, trialidx, tlabel, rtimes,cellidx::AbstractVector{Int64}=1:size(ppsth.counts,3);rtime_min=100.0,
                                                                        rtime_max=300.0,
                                                                        mean_subtract=false,
                                                                        variance_stabilize=false,kvs...)
	all_sessions = DPHT.get_level_path.("session", ppsth.cellnames[cellidx])
	sidx = findall(all_sessions.==session)
	if isempty(sidx)
		return nothing, nothing, nothing
	end
	_trialidx = trialidx[cellidx][sidx][1]
	_label = tlabel[cellidx][sidx][1]
	ntrials = length(_label)
	rtime = rtimes[session][_trialidx]
	rtidx = findall(rtime_min .< rtime .< rtime_max)
	rtime = rtime[rtidx]
	_label = _label[rtidx]
	X = ppsth.counts[:,rtidx,cellidx[sidx]]
    if variance_stabilize
        X .= sqrt.(X)
    end
    if mean_subtract
        X .-= mean(X, dims=(1,2))
    end
	X, _label, rtime
end

"""
```julia
function get_normals(traj::Matrix{T}) where T <: Real
````
Get the normals at each point of the trajectory.

A normal vector is a unit vector orthogonal to the trajectory gradient at each time point

"""
function get_normals(traj::Matrix{T}) where T <: Real
	#find the tangential vector at each point
	grad = diff(traj, dims=1)
	grad ./= sqrt.(sum(abs2, grad,dims=2))
	#find the vector orthogonal to the gradient
	nn = randn(size(grad)...)
	# remove projection onto grad
	nn .-= sum(nn.*grad,dims=2).*grad
	#re-normalize
	nn ./= sqrt.(sum(abs2,nn,dims=2))
	@debug "Test" extrema(sum(nn.*grad,dims=2))
	nn
end

function gaussian_smooth(X::AbstractVector{T}, bins::AbstractVector{T},σ::T) where T <: Real
    Δ = mean(diff(bins))
    Y = fill(0.0, length(X))
    for i in 1:length(bins)
        l = bins[end]-bins[i]
        u = bins[1]-bins[i]
        N = Truncated(Normal(0.0, σ),min(l,u) ,max(l,u))
        Y[i] = sum(pdf.(N, bins[i] .- bins).*X*Δ)
    end
    Y
end

function get_keyword_args(m::Method)
	ms = string(m)
	idx0 = findfirst(';', ms)
	idx1 = findlast(')', ms)
	sig = ms[idx0+1:idx1-1]
	vars = filter(s->!occursin("...", s), lstrip.(split(ms[idx0:idx1],",")))
end

function load_data(subject::Union{String,Nothing}=nothing;area="fef",align=:cue, raw=false,parallel_read=true, kvs...)
    if subject == "M"
        # this is model data
        fname = joinpath(@__DIR__, "..", "data","ppsth_model_cue.jld2")
    elseif raw
        fname = joinpath(@__DIR__, "..", "data","ppsth_$(area)_$(align)_raw.jld2")
    else 
        fname = joinpath(@__DIR__, "..", "data","ppsth_$(area)_$(align).jld2")
    end
    ppsth,tlabels,trialidx, rtimes = JLD2.load(fname, "ppsth","labels","trialidx","rtimes";parallel_read=parallel_read)
    return ppsth,tlabels,trialidx,rtimes
end

export get_area_index, get_session_data, rebin2, sessions_j, session_w, ncells, _ntrials, locations, location_mapping, location_idx, get_normals, load_data

end #module