module Figure5
using LinearAlgebra
using Random
using StatsBase

using Makie
using Colors
using CairoMakie

using ..Utils
using ..PlotUtils

target_colors = let
    colors = [RGB(0.8, 0.8, 0.8); parse.(Colorant, ["black","deepskyblue", "tomato"])]
    push!(colors, parse(Colorant, "white"))
    cc = distinguishable_colors(8, colors, dropseed=true)
    [RGB(0.5 + 0.5*tg.r, 0.5 + 0.5*tg.g, 0.5 + 0.5*tg.b) for tg in cc]
end

function plot_saccades!(ax, saccades::Vector{T2}, color, qcolor=color;xmin=0.0, xmax=1500.0, ymin=0.0, ymax=1000.0, max_q::Union{Vector{Int64}, Nothing}=nothing) where T2 <: Matrix{T} where T <: Real
    for (ii,saccade) in enumerate(saccades)
		if max_q === nothing
			x = saccade[:,1]
			y = saccade[:,2]

			idx = (xmin .< x .< xmax).&(ymin .< y .< ymax)
			lines!(ax, x[idx], y[idx], color=color)
		else
			eq = max_q[ii]
			x1 = saccade[1:eq,1]
			x2 = saccade[eq+1:end,1]
			y1 = saccade[1:eq,2]
			y2 = saccade[eq+1:end,2]
			for (x,y,_color) in zip([x1,x2], [y1,y2],[qcolor,color])
				idx = (xmin .< x .< xmax).&(ymin .< y .< ymax)
				lines!(ax, x[idx], y[idx], color=_color)
			end
		end
    end
end

function plot_microstimulation_figure!(figlg)
    # load saccade data for sessions with early stimulations
    _sdata_early = JLD2.load("data/microstim_early_sessions.jld2")
    sdata_early = NamedTuple(zip(Symbol.(keys(_sdata_early)), values(_sdata_early)))

    # .. and late stimulation
    _sdata_mid = JLD2.load("data/microstim_mid_sessions.jld2")
    sdata_mid = NamedTuple(zip(Symbol.(keys(_sdata_mid)), values(_sdata_mid)))

	pos0 = sdata_early.target_pos[findfirst(pos->(pos[1] > sdata_early.screen_center[1])&(pos[2] == sdata_early.screen_center[2]), sdata_early.target_pos)]

	qidxe = [pos==pos0 for pos in sdata_early.target_pos[sdata_early.trial_label_stim]]
	yidxe = [sqrt(sum(abs2,saccade[end,:] .- pos0)) < 2*sdata_early.target_size for saccade in sdata_early.saccades_stim]

	qidxm = [pos==pos0 for pos in sdata_mid.target_pos[sdata_mid.trial_label_stim]]
	yidxm = [sqrt(sum(abs2,saccade[end,:] .- pos0)) < 2*sdata_mid.target_size for saccade in sdata_mid.saccades_stim]

	# plot thre reacction times first
	rtime_nostim = [sdata_early.rtime_nostim; sdata_mid.rtime_nostim]

	# determine the lower threshold from the rtime_nostim
	rt_cutoff = percentile(rtime_nostim, 5)
	rt_cutoff = minimum(rtime_nostim)
	@debug rt_cutoff
	rtime_stim_early = sdata_early.rtime_stim
	rtime_stim_mid = sdata_mid.rtime_stim
	
	rtidx_nostim = fill(true, length(rtime_nostim))
	rtidx_stim_early = fill(true, length(rtime_stim_early))
	rtidx_stim_mid = fill(true, length(rtime_stim_mid))

	evoked_saccade_idx = (early=fill(false, length(rtime_stim_early)),
						  mid=fill(false, length(rtime_stim_mid)),
						  nostim=fill(false, length(rtime_nostim)))
	
	fast_saccade_idx = (early=fill(false, length(rtime_stim_early)),
						  mid=fill(false, length(rtime_stim_mid)),
						  nostim=fill(false, length(rtime_nostim)))
	
	# compute saccade length so that we can filter out long artifacts
	slength_stim_early = fill(0.0, length(sdata_early.saccades_stim))
	slength_stim_mid = fill(0.0, length(sdata_mid.saccades_stim))
	slength_nostim = fill(0.0, length(rtime_nostim))
	max_q_idx_early = fill(0, length(slength_stim_early))
	max_q_idx_mid = fill(0, length(slength_stim_mid))
	max_q_idx_nostim = fill(0, length(slength_nostim))

	# exclude the middle right position since this was where all of the evoked saccade went.
	pos0 = sdata_early.target_pos[findfirst(pos->(pos[1] > sdata_early.screen_center[1])&(pos[2] == sdata_early.screen_center[2]), sdata_early.target_pos)]

	pos0v = fill(0.0, 1,2)
	pos0v[1,:] .= pos0
	all_target_pos = sdata_mid.target_pos
	# get the distance from the central fixation to each of the target locations
	tvector = [pos .- sdata_mid.screen_center for pos in sdata_mid.target_pos]
	
	for (q, slength, max_q, saccades, trial_label,rtidx, target_pos) in zip([:early, :mid, :nostim], [slength_stim_early, slength_stim_mid, slength_nostim], [max_q_idx_early, max_q_idx_mid, max_q_idx_nostim], [sdata_early.saccades_stim, sdata_mid.saccades_stim, [sdata_early.saccades_nostim;sdata_mid.saccades_nostim]], [sdata_early.trial_label_stim, sdata_mid.trial_label_stim, [sdata_early.trial_label_nostim;sdata_mid.trial_label_nostim]], [rtidx_stim_early, rtidx_stim_mid, rtidx_nostim], [sdata_early.trial_label_stim, sdata_mid.trial_label_stim, [sdata_early.trial_label_nostim;sdata_mid.trial_label_nostim]])
		for (jj,saccade) in enumerate(saccades)
			v = sqrt.(sum(diff(saccade, dims=1).^2,dims=2))
			w = diff(saccade, dims=1)
			w ./= v  #normalize

			v = dropdims(v, dims=2)
			# find the largest directional change
			# max_q[jj] = argmin(dropdims(sum(w[1:end-1,:].*w[2:end,:],dims=2),dims=2)) .+ 1
			# use the positions closest to pos0 as max_q_idx
			dd = dropdims(sum(abs2, saccade .- pos0v, dims=2),dims=2)
			max_q[jj] = argmin(dd)
			slength[jj] = sum(v)
			tlength = sqrt(sum(tvector[trial_label[jj]].^2))
			slength[jj] /= tlength
		end
		# only keep short saccades
		rtidx[slength .> 1.4] .= false
		# identify saccades that go to the middle right location
		qidx = [pos==pos0 for pos in all_target_pos[target_pos]]
		# identity saccades where the end point is close to the middle right location
		yidx = [sqrt(sum(abs2,saccade[end,:] .- pos0)) < 2*sdata_mid.target_size for saccade in saccades]
		evoked_saccade_idx[q][(qidx .| yidx).|(slength .> 1.4)] .= true
		fast_saccade_idx[q][((!).(qidx .| yidx)).&(slength .< 1.4)] .= true
		rtidx[qidx] .= false
		rtidx[yidx] .= false
	end

	

	# find the indices of this 

	with_theme(theme_minimal()) do
		ax1 = Axis(figlg[1,1])
		ax1.title = "No stim"
		ax1.yticksvisible = true
		ax1.ylabel = "Reaction time [ms]"
		scatter!(ax1, rand(sum(rtidx_nostim)), rtime_nostim[rtidx_nostim], color=RGB(0.8, 0.8, 0.8))
		ax2 = Axis(figlg[1,2])
		ax2.title = "Early stim"
		scatter!(ax2, rand(sum(rtidx_stim_early)), rtime_stim_early[rtidx_stim_early], color=Cycled(3))
		ax3 = Axis(figlg[1,3])
		ax3.title = "Late stim"
		scatter!(ax3, rand(sum(rtidx_stim_mid)), rtime_stim_mid[rtidx_stim_mid], color=Cycled(4))
		# plot histogras
		ax6 = Axis(figlg[1,4])
		ax6.yticklabelsvisible = false
		ax6.leftspinevisible = false
		ax6.xticksvisible = true
		ax6.xlabel = "% of trials"
		# kind of fudgy
		binsize = rt_cutoff - minimum(rtime_stim_mid[rtidx_stim_mid])
		binsize /= 1.0
		@debug binsize
		firstbin = minimum(minimum.([rtime_stim_mid[rtidx_stim_mid],rtime_stim_early[rtidx_stim_early], rtime_nostim[rtidx_nostim]]))
		lastbin = maximum(maximum.([rtime_stim_mid[rtidx_stim_mid],rtime_stim_early[rtidx_stim_early], rtime_nostim[rtidx_nostim]]))
		bins = range(firstbin-binsize, stop=lastbin, step=binsize)
		h1 = StatsBase.fit(Histogram, rtime_nostim[rtidx_nostim], bins, closed=:left)
		h2 = StatsBase.fit(Histogram, rtime_stim_early[rtidx_stim_early], bins, closed=:left)
		h3 = StatsBase.fit(Histogram, rtime_stim_mid[rtidx_stim_mid], bins, closed=:left)	

		#barplot!(ax6, h1.edges[1][1:end-1], 100*h1.weights/sum(h1.weights), color=RGB(0.8, 0.8, 0.8),width=binsize, gap=0.0, direction=:x)
		stairs!(ax6, 100*h1.weights/sum(h1.weights), h1.edges[1][1:end-1], color=RGB(0.8, 0.8, 0.8))
		#barplot!(ax6, h2.edges[1][1:end-1], 100*h2.weights/sum(h2.weights), color=ax6.palette.color[][3], width=binsize, gap=0.0, direction=:x)
		stairs!(ax6, 100*h2.weights/sum(h2.weights), h2.edges[1][1:end-1], color=ax6.palette.color[][3])
		#barplot!(ax6, h3.edges[1][1:end-1], 100*h3.weights/sum(h3.weights), color=ax6.palette.color[][4], width=binsize, gap=0.0, direction=:x)
		stairs!(ax6, 100*h3.weights/sum(h3.weights), h3.edges[1][1:end-1], color=ax6.palette.color[][4])
		hlines!(ax6, rt_cutoff, linestyle=:dot, color="black")
		linkyaxes!(ax1, ax2, ax3,ax6)
		for ax in [ax2, ax3]
			ax.leftspinevisible = false
			ax.yticksvisible = false
			ax.yticklabelsvisible = false
		end
		for ax in [ax1, ax2, ax3]
			ax.xticklabelsvisible = false
			hlines!(ax, rt_cutoff, linestyle=:dot, color="black")
		end

		# plot the saccades, too
		lg = GridLayout()
		figlg[2,1:4] = lg

		ax4 = Axis(lg[1,1], xticklabelsvisible=false, yticklabelsvisible=false)
		ax5 = Axis(lg[1,2], xticklabelsvisible=false, yticklabelsvisible=false)

		# only show the saccade traces for fast saccades
		rtidx_stim_early .= (rtime_stim_early .< rt_cutoff).&fast_saccade_idx.early
		rtidx_stim_mid .= (rtime_stim_mid .< rt_cutoff).&fast_saccade_idx.mid
		
		for (ax,rtidx, saccades, color) in zip([ax4, ax5], [rtidx_stim_early, rtidx_stim_mid], [sdata_early.saccades_stim, sdata_mid.saccades_stim], ax6.palette.color[][3:4])
			scatter!(ax, [pos[1] for pos in sdata_early.target_pos], [pos[2] for pos in sdata_early.target_pos], marker='□', markersize=sdata_early.target_size, color=target_colors, markerspace=:data)
			plot_saccades!(ax, saccades[rtidx], color)
			xlims!(ax, 300.0, 1600.0)
			ylims!(ax, 0.0, 1200.0)
		end
		ax4.ylabel = "Fast saccades"

		ax7 = Axis(lg[2,1], xticklabelsvisible=false, yticklabelsvisible=false)
		ax8 = Axis(lg[2,2], xticklabelsvisible=false, yticklabelsvisible=false)

		# only show the saccade traces for evoked saccades
		rtidx_stim_early .= (rtime_stim_early .< rt_cutoff).&evoked_saccade_idx.early
		rtidx_stim_mid .= (rtime_stim_mid .< rt_cutoff).&evoked_saccade_idx.mid

		@debug extrema(slength_stim_mid)
		@debug sum(evoked_saccade_idx.mid)
		@show max_q_idx_early
		for (ax,rtidx, saccades, max_q, color, qcolor) in zip([ax7, ax8], [rtidx_stim_early, rtidx_stim_mid], [sdata_early.saccades_stim, sdata_mid.saccades_stim],[max_q_idx_early, max_q_idx_mid],  ax6.palette.color[][[3,5]], ax6.palette.color[][[4,6]])
			scatter!(ax, [pos[1] for pos in sdata_early.target_pos], [pos[2] for pos in sdata_early.target_pos], marker='□', markersize=sdata_early.target_size, color=target_colors, markerspace=:data)
			plot_saccades!(ax, saccades[rtidx], color,qcolor;max_q=max_q[rtidx])
			xlims!(ax, 300.0, 1600.0)
			ylims!(ax, 0.0, 1200.0)
		end
		ax7.ylabel = "Evoked saccades"
		#plot_saccades!(ax4, )
		rowsize!(lg, 1, Relative(0.4))
    end
end

function plot()
	cue_color = RGB(0.7, 1.0, 0.7)
	saccade_color =RGB(1.0, 0.676, 0.3) 
	with_theme(plot_theme) do
		fig = Figure(resolution=(700,900))
		lg1 = GridLayout()
		fig[1,1] = lg1
		# drawing to illustrate stimulation
		ax = Axis(lg1[1,1], xticksvisible=false, xticklabelsvisible=false, yticksvisible=false, yticklabelsvisible=false, bottomspinevisible=false, leftspinevisible=false)
		ax2 = Axis(lg1[2,1], xticksvisible=false, xticklabelsvisible=false, yticksvisible=false, yticklabelsvisible=false, bottomspinevisible=false, leftspinevisible=false)
		linkaxes!(ax,ax2)
		rowsize!(lg1, 2, Relative(0.2))
		rowgap!(lg1, 1, 0.0)
		arrows!(ax, [0.0], [0.0], [1.0], [0.0],linestyle=:dot)
		poly!(ax, Rect2(0.1, 0.0, 0.05, 0.3), color=cue_color)
		poly!(ax, Rect2(0.3, 0.0, 0.1, 0.29), color=:white, strokewidth=1.0)
		text!(ax, 0.35, 0.15, text="25ms", align=(:center, :center))
		# lightning bolt
		scatter!(ax, [0.2], [0.4], marker='⚡', color=RGB(1.0, 0.9, 0.0), markersize=25px)
		poly!(ax, Rect2(0.8, 0.0, 0.05, 0.3), color=saccade_color)
		poly!(ax, Rect2(0.65, 0.0, 0.15, 0.29), color=:white, strokewidth=1.0)
		text!(ax, 0.725, 0.15, text="35ms", align=(:center, :center))
		ll = text!(ax2, [0.1, 0.8],[0.0, 0.0], text=["Go cue","Saccade"], color=[cue_color, saccade_color],fontsize=12)
		ylims!(ax2, -0.1, 0.45)
		lg2 = GridLayout()
		fig[2,1] = lg2
		plot_microstimulation_figure!(lg2)
		rowsize!(fig.layout, 1, Relative(0.1))
		fig
		
	end
end
end
