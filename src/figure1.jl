module Figure1
using Makie
using Makie: Point2f0
using CairoMakie
using JLD2
using Colors
using FileIO
using CRC32c
using HDF5
using Random
using ProgressMeter
using StatsBase
using MultivariateStats
using EventOnsetDecoding
using DataProcessingHierarchyTools

const DPHT = DataProcessingHierarchyTools

using ..Utils
using ..PlotUtils

CairoMakie.activate!()
"""
Create an illustration of the task into the layout `lg`
"""
function task_figure!(lg;kvs...)
    fontsize = get(kvs, :fontsize, 12)
    #task for James/Pancake
	lg1 = GridLayout()
	lg[1,1] = lg1
	axes = [Axis(lg1[1,i],xticksvisible=false, yticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false,backgroundcolor=RGB(0.8, 0.8, 0.8),leftspinevisible=false,bottomspinevisible=false,aspect=1,xlabelsize=fontsize) for i in 1:7]
	# grid
	labels = ["Fixation\n500ms","Target\n300ms","Delay 1\n1000ms","Distractor\n300ms","Delay 2\n1000ms", "Go-cue","Saccade"]
	for (ii,ax) in enumerate(axes)
		make_grid!(ax)	
		if ii <= 5 
			poly!(ax, Circle(Point2f0(1.5,1.5), 0.1),color="white")
		elseif ii == 7
			arrows!(ax, [1.5], [1.5], [1.0], [0.0])
		end
		if ii == 2
			poly!(ax, Rect2(Point2f0(2.0, 1.0), Point2f0(1.0, 1.0)), color="red")
		elseif ii == 4
			poly!(ax, Rect2(Point2f0(0.0, 0.0), Point2f0(1.0, 1.0)), color="green")
		end
		ax.xlabel = labels[ii]
	end
	labels2 = ["Fixation\n500ms","Target\n300ms","Delay 1\n1000ms","Distractor\nor re-target\n300ms","Delay 2\n1000ms", "Go-cue","Saccade"]
	lg2 = GridLayout()
	lg[2,1] = lg2
	lg21 = GridLayout()
	lg2[1,1] = lg21
	lg22 = GridLayout()
	lg2[1,2] = lg22
	axes2 = [Axis(lg21[1,i],xticksvisible=false, yticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false,backgroundcolor=RGB(0.8, 0.8, 0.8),leftspinevisible=false,bottomspinevisible=false,aspect=1,xlabelsize=fontsize) for i in 1:3]
	axes3 = [Axis(lg22[1,i],xticksvisible=false, yticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false,backgroundcolor=RGB(0.8, 0.8, 0.8),leftspinevisible=false,bottomspinevisible=false,aspect=1,xlabelsize=fontsize) for i in 1:4]
	axes4 = [Axis(lg22[3,i],xticksvisible=false, yticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false,backgroundcolor=RGB(0.8, 0.8, 0.8),leftspinevisible=false,bottomspinevisible=false, aspect=1, xlabelsize=fontsize) for i in 1:4]
	rowsize!(lg21,1,Relative(0.5))
	colsize!(lg2, 1, Relative(3/7.0))
	for (ii,_ax) in enumerate(axes2)
		poly!(_ax, Circle(Point2f0(1.5,1.5), 0.1),color="white")
		if ii == 2
			poly!(_ax, Rect2(Point2f0(2.0, 2.0), Point2f0(1.0, 1.0)), color="red")
		end
		_ax.xlabel = labels2[ii]
	end
	for (ii,_ax) in enumerate(axes3)
		if ii <= 2
			poly!(_ax, Circle(Point2f0(1.5,1.5), 0.1),color="white")	
		end
		if ii == 1
			poly!(_ax, Rect2(Point2f0(0.0, 0.0), Point2f0(1.0, 1.0)), color="green")
		elseif ii == 4
			aa = arrows!(_ax, [1.5], [1.5], [1.0], [1.0])
			translate!(aa, 0.0, 0.0, 1.0)
		end
	end
	for (ii,_ax) in enumerate(axes4)
		if ii <= 2
				poly!(_ax, Circle(Point2f0(1.5,1.5), 0.1),color="white")	
		end
		if ii == 1
			poly!(_ax, Rect2(Point2f0(0.0, 0.0), Point2f0(1.0, 1.0)), color="red")
		elseif ii == 4
			aa = arrows!(_ax, [1.5], [1.5], [-1.0], [-1.0])
			translate!(aa, 0.0, 0.0, 1.0)
		end
	end
	for _ax in [axes2;axes3;axes4]
		scatter!(_ax, [0.5,2.5,0.5,2.5],[0.5, 0.5, 2.5, 2.5],markersize=1.5, marker='+',color="white",markerspace=:data)
		xlims!(_ax, -0.1, 3.1)
		ylims!(_ax, -0.1, 3.1)
	end
    axes[1,1].title = "Monkey J"
    axes2[1,1].title = "Monkey W"
	_ll = [Label(lg22[2,i], labels2[3+i],tellwidth=false,fontsize=fontsize) for i in 1:4]
	rowgap!(lg,1,0.0)
	rowsize!(lg, 1, Relative(1.0/3.5))
end

function plot(;do_save=true)
    img = load(joinpath("figures","manuscript","electrodes_on_brain.png"))
	# grap the width and heigth from the QUARTO ENV variable
	width = parse(Float64, get(ENV, "QUARTO_FIG_WIDTH", "15"))*72
	height = parse(Float64, get(ENV, "QUARTO_FIG_HEIGHT","15"))*72
	with_theme(plot_theme) do
		fig = Figure(size=(width,height))
		lg1 = GridLayout()
		fig[1,1] = lg1
		task_figure!(lg1;fontsize=16)
		# TODO: Add the implant figure here. Just load it from a file
		lg2 = GridLayout()
		fig[2,1] = lg2
		ax = Axis(lg2[1,1], xticksvisible=false, yticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false,leftspinevisible=false, bottomspinevisible=false,aspect=DataAspect())
		image!(ax,rotr90(img))
		rowsize!(fig.layout, 2, 442)

		#add labels
		label_fontsize = 24 
		labels = [Label(lg1[1,1,TopLeft()], "A",fontsize=label_fontsize),
				  Label(lg2[1,1,TopLeft()], "B",fontsize=label_fontsize)]
		fname = joinpath("figures","manuscript","figure1.pdf")
        if do_save
            CairoMakie.activate!()
            save(fname, fig;pt_per_unit=1)
        end
		fig
	end
end
end
