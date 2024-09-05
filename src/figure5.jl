module Figure5
using LinearAlgebra
using Random
using StatsBase
using JLD2

using Makie
using Colors
using CairoMakie
using HypothesisTests
using Distributions

using ..Utils
using ..PlotUtils

target_colors = let
    colors = [RGB(0.8, 0.8, 0.8); parse.(Colorant, ["black","deepskyblue", "tomato"])]
    push!(colors, parse(Colorant, "white"))
    cc = distinguishable_colors(8, colors, dropseed=true)
    [RGB(0.5 + 0.5*tg.r, 0.5 + 0.5*tg.g, 0.5 + 0.5*tg.b) for tg in cc]
end

"""
Find the number of points below q1 and above q2 in surrogates where rt1 and rt2 is mixed
"""
function shuffle_stim_rtimes(rt1, rt2, q1, q2;nruns=1000)
	rt = [rt1;rt2]
	n1 = length(rt1)
	n2 = length(rt2)
	nq = fill(0,2,2,nruns)
	for i in 1:nruns
		shuffle!(rt)
		_rt1 = rt[1:n1]
		_rt2 = rt[n1+1:end]
		nq[1,1,i] = sum(_rt1 .<= q1)
		nq[2,1,i] = sum(_rt1 .>= q2)
		nq[1,2,i] = sum(_rt2 .<= q1)
		nq[2,2,i] = sum(_rt2 .>= q2)
	end
	nq
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
	#rt_cutoff = minimum(rtime_nostim)
	@show rt_cutoff
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

	
	# statistics
	_rt_nostim = rtime_nostim[rtidx_nostim]
	_rt_early = rtime_stim_early[rtidx_stim_early]	
	_rt_mid = rtime_stim_mid[rtidx_stim_mid]
	@show HypothesisTests.KruskalWallisTest(_rt_nostim, _rt_early, _rt_mid)
	#@show HypothesisTests.ExactMannWhitneyUTest(_rt_early, _rt_mid)
	le,me,ue = percentile(_rt_early, [5,50,95])
	lm,mm,um = percentile(_rt_mid, [5,50,95])
	@show le, um
	@show lm, ue

	#fit Gamma distribution to nostim reaction times
	Γ = fit(Gamma, _rt_nostim)
	@show Γ
	# compare number of points below 1st percentile 
	q1,q2 = percentile(Γ, [5.0,95.0])
	@show q1,q2
	rt_cutoff = q1
	n11 = sum(_rt_early .<= q1)
	n12 = sum(_rt_mid .<= q1)
	n21 = sum(_rt_early .>= q2)
	n22 = sum(_rt_mid .>= q2)
	nq = shuffle_stim_rtimes(_rt_early, _rt_mid, q1,q2)
	nqs = mapslices(x->percentile(x,95), nq,dims=3)
	@assert !(n11 >= nqs[1,1])
	@assert !(n21 >= nqs[2,1])
	@assert n12 >= nqs[1,2] # this should be the only significant result
	@assert !(n22 >= nqs[2,2])
    plot_colors = Makie.wong_colors()
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
		for _ax in [ax1, ax6]
			@show median(rtime_nostim[rtidx_nostim])
			hlines!(_ax, median(rtime_nostim[rtidx_nostim]), color=RGB(0.8, 0.8, 0.8))
		end
		stairs!(ax6, 100*h2.weights/sum(h2.weights), h2.edges[1][1:end-1], color=plot_colors[3])
		for _ax in [ax2, ax6]
			@show median(rtime_stim_early[rtidx_stim_early])
			hlines!(_ax, median(rtime_stim_early[rtidx_stim_early]), color=plot_colors[3])
		end
		stairs!(ax6, 100*h3.weights/sum(h3.weights), h3.edges[1][1:end-1], color=plot_colors[4])
		for _ax in [ax3, ax6]
			@show median(rtime_stim_mid[rtidx_stim_mid])
			hlines!(_ax, median(rtime_stim_mid[rtidx_stim_mid]), color=plot_colors[4])
		end
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
		
		for (ax,rtidx, saccades, color) in zip([ax4, ax5], [rtidx_stim_early, rtidx_stim_mid], [sdata_early.saccades_stim, sdata_mid.saccades_stim], plot_colors[3:4])
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
		for (ax,rtidx, saccades, max_q, color, qcolor) in zip([ax7, ax8], [rtidx_stim_early, rtidx_stim_mid], [sdata_early.saccades_stim, sdata_mid.saccades_stim],[max_q_idx_early, max_q_idx_mid],  plot_colors[[3,5]], plot_colors[[4,6]])
			scatter!(ax, [pos[1] for pos in sdata_early.target_pos], [pos[2] for pos in sdata_early.target_pos], marker='□', markersize=sdata_early.target_size, color=target_colors, markerspace=:data)
			plot_saccades!(ax, saccades[rtidx], color,qcolor;max_q=max_q[rtidx])
			xlims!(ax, 300.0, 1600.0)
			ylims!(ax, 0.0, 1200.0)
		end
		ax7.ylabel = "Evoked saccades"
		for ax in [ax4,ax5, ax7, ax8]
			ax.xticksvisible = false
			ax.yticksvisible = false
			ax.rightspinevisible = true
			ax.topspinevisible = true
		end
		#plot_saccades!(ax4, )
		#rowsize!(lg, 1, Relative(0.4))
    end
end

function plot_schematic()
	fig = Figure(size=(450,210))
	bbox = BBox(5, 445, 5, 205)
	plot_schematic(fig, bbox)
	fig
end

function plot_schematic(fig,bbox)
	cue_color = :green
	saccade_color =RGB(1.0, 0.0, 0.0) 
	with_theme(plot_theme) do
		# drawing to illustrate stimulation
		factor = 1.15
		go_cue_onset = 0.0
		go_cue_width = 5.0*factor
		mp_onset = go_cue_width
		mp_width = 40.0*factor
		go_cue_signal_onset = go_cue_width+mp_width
		go_cue_signal_width = 15.0*factor
		transition_period_onset = go_cue_signal_onset+go_cue_signal_width
		transition_period_width=50.0*factor
		execution_threshold_onset = transition_period_onset+transition_period_width
		execution_threshold_width=20.0*factor
		saccade_onset = execution_threshold_onset+execution_threshold_width
		saccade_width = 5.0*factor
		total_length = go_cue_width + mp_width+go_cue_signal_width+transition_period_width
		total_length += execution_threshold_width
		total_length += saccade_width
		box_height = 0.3

		ax = Axis(fig, xticksvisible=false, xticklabelsvisible=false, yticksvisible=false, yticklabelsvisible=false, bottomspinevisible=false, leftspinevisible=false,bbox=bbox)
		ylims!(ax, -0.15, 0.75)
		arrows!(ax, [0.0], [-0.01], [total_length], [0.0],linewidth=2.0)
		poly!(ax, Rect2(0.0, 0.0, go_cue_width, 0.3), color=cue_color)

		offset = 0.17
		width = 0.22
		#poly!(ax, Rect2(offset, 0.0, width, 0.29), color=:white, strokewidth=0.0)
		text!(ax, mp_onset+0.5*mp_width, box_height/2, text=rich(rich("40 ms",font=:bold), rich("\nMotor \nPreparation")), align=(:center, :center), color=:black)

		# label for stimulation
		label_color = :yellow
		yoffset = 0.32
		height = 0.23
		# this should end at the beginning of the go-cue
		poly!(ax, Rect2(mp_onset+5.0*factor, yoffset, 50.0*factor, height), color=label_color, strokewidth=1.0)
		text!(ax, mp_onset+5.0*factor + 25.0*factor, yoffset+height/2, text="5-55 ms\n(early stim)", align=(:center, :center))

		_offset = offset + 0.9*width
		_width = 1.1*width
		poly!(ax, Rect2(40.0*factor, yoffset+0.2, 50.0*factor, height), color=label_color, strokewidth=1.0)
		text!(ax, (40.0+25.0)*factor, yoffset+0.2+height/2, text="40-90 ms\n(late stim)", align=(:center, :center))

		offset = offset+width
		width = 0.16
		poly!(ax, Rect2(go_cue_signal_onset, 0.0, go_cue_signal_width, box_height), color=:white, strokewidth=2.0)
		text!(ax, go_cue_signal_onset + 0.5*go_cue_signal_width, 0.15, text=rich(rich("15 ms", font=:bold),rich("\n Go Cue\nSignal")), align=(:center, :center),color=cue_color)
		# lightning bolt
		#scatter!(ax, [0.2], [0.4], marker='⚡', color=RGB(1.0, 0.9, 0.0), markersize=25px)

		offset = offset + width
		width = 0.18
		text!(ax, transition_period_onset+0.5*transition_period_width, 0.15, text="Transition\n Period", align=(:center, :center),color=:black)

		offset = offset + width
		width = 0.20
		poly!(ax, Rect2(execution_threshold_onset, 0.0, execution_threshold_width, box_height), color=:white, strokewidth=2.0)
		text!(ax, execution_threshold_onset+0.5*execution_threshold_width, 0.15, text=rich(rich("35 ms",font=:bold), rich("\nExecution\nThreshold")), align=(:center, :center),color=saccade_color)

		offset = offset + width
		width = 0.05
		poly!(ax, Rect2(saccade_onset, 0.0, saccade_width, box_height), color=saccade_color)

		text!(ax, [2.5,saccade_onset+0.5*saccade_width],[-0.06, -0.06], text=["Go cue","Saccade"], color=[cue_color, saccade_color], align=(:center, :center))
	end
end

function plot(;show_schematic=true)
	cue_color = RGB(0.7, 1.0, 0.7)
	saccade_color =RGB(1.0, 0.676, 0.3) 
	with_theme(plot_theme) do
		fwidth = 700
		fheight = 900
		fig = Figure(size=(fwidth, fheight))
		lg1 = GridLayout()
		fig[1,1] = lg1
		if show_schematic
			hh = 180
			rowsize!(fig.layout, 1, hh)
			_top = fheight-5
			bbox = BBox(75, fwidth-75, _top-hh, _top) 
			plot_schematic(fig, bbox)
			lg2 = GridLayout()
			fig[2,1] = lg2
			labels = [Label(fig[1,1,TopLeft()], "A",font=:regular),
					  Label(lg2[1,1,TopLeft()], "B", font=:regular),
					  Label(lg2[2,1,TopLeft()], "C", font=:regular)
					 ]
		else
			lg2 = lg1
		end
		plot_microstimulation_figure!(lg2)
		fig
	end
end

function StatsBase.loglikelihood(gm::GMM, x)
	N = [Normal(μ, σ) for (μ, σ) in zip(gm.μ, gm.Σ)]
	sum(log.(sum(gm.w.*pdf.(N, permutedims(x)),dims=1)))
end

function StatsBase.bic(gm::GMM, x)
	ll = loglikelihood(gm,x)
	k = length(gm.μ) + length(gm.Σ) + length(gm.w)-1
	n = length(x)
	-2*ll + k*log(n)
end

function plot_bimodal_analysis()

	# load saccade data for sessions with early stimulations
    _sdata_early = JLD2.load("data/microstim_early_sessions.jld2")
    # .. and late stimulation
    _sdata_mid = JLD2.load("data/microstim_mid_sessions.jld2")

	rtime_early = _sdata_early["rtime_stim"]
	rtime_mid = _sdata_mid["rtime_stim"]
	rtime_nostim = [_sdata_early["rtime_nostim"];_sdata_mid["rtime_nostim"]]

	fname = joinpath("data","figure5_bimodel_analysis.jld2")
	if isfile(fname)
		model_bic, n_modes, mode_assignment, model_converged,model = JLD2.load(fname, "model_bic",
																			    "n_modes",
																				"mode_assignment",
																				"model_converged",
																				"model")
	else
		model_bic = Dict{String, Vector{Float64}}()
		n_modes = Dict{String, Vector{Int64}}()
		mode_assignment = Dict{String, Vector{Int64}}()
		model_converged = Dict{String, Vector{Bool}}()
		model = Dict{String, Any}()
		for (ll,x) in zip(["nostim", "early","mid"],[rtime_nostim, rtime_early, rtime_mid])
			gm_3,_ = fit(GammaMixture, x, 3;niter=20_000)
			bic_3 = bic(gm_3, x)
			gm_2,_ = fit(GammaMixture, x, 2;niter=20_000)
			bic_2 = bic(gm_2, x)
			g = fit(Gamma, x)
			bic_1 = -2*sum(logpdf.(g, x)) + 2*log(length(x))
			best_bic, best_model_idx = findmin([bic_3, bic_2, bic_1])
			best_model = [gm_3,gm_2,g][best_model_idx]

			probs = posterior(best_model, x)
			zq = [argmax(probs[i,:]) for i in axes(probs,1)]
			n_modes[ll] = [3,2,1]
			model_bic[ll] = [bic_3, bic_2, bic_1]
			mode_assignment[ll] = zq
			model_converged[ll] = [gm_3.converged, gm_2.converged, true]
			model[ll] = best_model
		end
		JLD2.save(fname, Dict("model_bic"=>model_bic, "n_modes"=>n_modes,"mode_assignment"=>mode_assignment,
						     "model_converged"=>model_converged,"model"=>model))
	end

	@show model_converged
	# create plots of BIC for each model, for early and mid stimulation
	colors = [to_color(:gray); Makie.wong_colors()[3:4]]
	with_theme(plot_theme) do
		fig = Figure()
		axes = [Axis(fig[1,i]) for i in 1:3]
		for (ll,ax,cc) in zip(["nostim","early","mid"], axes,colors[1:3])
			barplot!(ax, [1:3;], model_bic[ll],color=cc)
			best_bic, best_model_idx = findmin(model_bic[ll])
			annotations!(ax, ["*"], [Point2f(best_model_idx, best_bic)])
			# adjust the y-limits so that the annotations can be seen
			mii,mxx = extrema(model_bic[ll])
			ymax = 1.1*mxx
			ylims!(ax, 0.0, ymax)
			ax.xticks = ([1:3;],string.([3,2,1]))
			ax.xlabel = "# modes"
		end
		axes[1].ylabel = "BIC"

		lg2 = [GridLayout(fig[2,i]) for i in 1:3]
		for (ll,x, lg,cc) in zip(["nostim","early","mid"], [rtime_nostim, rtime_early, rtime_mid],lg2,colors)
			# make one color a little brighter, one a little darker
			q = 0.7
			p = 1.0 -q
			cc0 = RGB(q*cc.r, q*cc.g, q*cc.b)
			cc1 = RGB(p + q*cc.r, p+q*cc.g, p+q*cc.b)
			_colors = [cc0, cc1]
			ax1 = Axis(lg[1,1])
			ax2 = Axis(lg[1,2])
			linkyaxes!(ax1, ax2)
			best_model = model[ll]
			xs = sort(x)
			probs = component_pdf(best_model, xs)
			for c in 1:size(probs,2)
				lines!(ax2, probs[:,c], xs,color=_colors[c])
			end
			scatter!(ax1, rand(length(x)), x,color=_colors[mode_assignment[ll]])
			colsize!(lg, 1, Relative(0.7))
			ax2.yticksvisible = false
			ax2.yticklabelsvisible = false
			ax2.xticklabelsvisible = false
			ax2.xlabel = "pdf"
			ax1.xticksvisible = false
			ax1.xticklabelsvisible = false
			# TODO: Plot the distributions
		end
		fig
	end
end

end
