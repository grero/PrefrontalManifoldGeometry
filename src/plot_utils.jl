module PlotUtils
using Makie
using Makie: Point2f
using Colors

using ..Utils

plot_theme = Theme(Axis=(xticksvisible=true,
	            yticksvisible=true,
				xgridvisible=false,
				ygridvisible=false,
				topspinevisible=false,
				rightspinevisible=false,
				xticklabelsize=14,
				yticklabelsize=14))


function make_grid!(ax)
    linesegments!(ax, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],[0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0],color="black")
    linesegments!(ax, [0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0],[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],color="black")
end

function get_rectangular_border(x0,y0, x1,y1)
	[Point2f(x0, y0), Point2f(x0, y1), Point2f(x1, y1), Point2f(x1, y0), Point2f(x0, y0)]
end

"""
```julia
function advance(traj::Matrix,i::Int)
```
Return the next point in the trajectory from index `i`
"""
function advance(traj::Matrix,i::Int, Δ::Int)
	j = i+Δ
	if j >= size(traj,1)
		return nothing, i
	end
	traj[j,:], j
end

"""
```julia
function advance(traj::Matrix, i::Int, Δ::Real)
```
Return the point in the trajectory obtained by advancing an amount `Δ`
along the trajectory from point `i`
"""
function advance(traj::Matrix, i::Int, Δ::Float64)
	d = 0.0
	j = i+1
	while j < size(traj,1)
		d += sqrt.(sum(abs2,traj[j,:] - traj[j-1,:]))
		if d >= Δ
			break
		end
		j += 1
	end	
	if d == 0.0
		return nothing, i
	end
	traj[j,:],j
end

"""
```julia
function plot_normals!(ax, traj::Matrix{T}) where T <: Real
```
Plot the normals at each point in the trajectory.
"""
function plot_normals!(ax, traj::Matrix{T};ss::Function=(traj,i)->advance(traj,i,1), linelength=0.08,kvs...) where T <: Real
	ll = linelength/2
	nn = get_normals(traj)
	lower = Vector{T}[]
	upper = Vector{T}[]
	j = 1
	while j < size(traj,1)
		p0,j = ss(traj,j)
		if p0 != nothing && j < size(traj,1)
			push!(lower, p0-ll*nn[j,:])
			push!(upper, p0+ll*nn[j,:])
		else 
			break
		end
	end
	upper = permutedims(cat(upper...,dims=2), [2,1])
	lower = permutedims(cat(lower...,dims=2), [2,1])
	#lower = traj[1:end-1,:] - ll*nn
	#upper = traj[1:end-1,:] + ll*nn
	xlower_upper = permutedims([lower[:,1] upper[:,1]], [2,1])[:]
	ylower_upper = permutedims([lower[:,2] upper[:,2]], [2,1])[:]
	linesegments!(ax, xlower_upper, ylower_upper;kvs...)
end

export plot_theme, make_grid!, get_rectangular_border, plot_normals!, advance
end
