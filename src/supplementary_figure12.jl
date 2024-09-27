module FigureS12
using JD2
using GammaMixtures
using CairoMakie
using Distributions
using StatsBase

using ..Utils
using ..Figure5 

function plot(;redo=false, nruns=100,plot_loglikelihood=false)

	# load saccade data for sessions with early stimulations
    _sdata_early = JLD2.load("data/microstim_early_sessions.jld2")
	sdata_early = NamedTuple(zip(Symbol.(keys(_sdata_early)), values(_sdata_early)))
    # .. and late stimulation
    _sdata_mid = JLD2.load("data/microstim_mid_sessions.jld2")
	sdata_mid = NamedTuple(zip(Symbol.(keys(_sdata_mid)), values(_sdata_mid)))

	classified_saccades_mid = Figure5.classify_saccades(sdata_mid)
	classified_saccades_early = Figure5.classify_saccades(sdata_early)
	rtime_early = _sdata_early["rtime_stim"][classified_saccades_early.rtidx.stim]
	rtime_mid = _sdata_mid["rtime_stim"][classified_saccades_mid.rtidx.stim]
	rtime_nostim = [_sdata_early["rtime_nostim"][classified_saccades_early.rtidx.nostim];_sdata_mid["rtime_nostim"][classified_saccades_mid.rtidx.nostim]]

	fname = joinpath(@__DIR__, "..", "data","figure5_bimodel_analysis.jld2")
	if !redo && isfile(fname)
		model_bic, n_modes, mode_assignment, model_converged,model,loglike,loglike_nostim = JLD2.load(fname, "model_bic",
																			    "n_modes",
																				"mode_assignment",
																				"model_converged",
																				"model",
																				"loglike",
																				"loglike_nostim")
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
		# to check whether the overall distribution has changed, compute
		# the loglikelihood of the reaction times from the stimulation trials using the 
		# model fit to the non-stimulation trials 
		loglike = Dict{String, Vector{Float64}}()
		loglike_nostim = Dict{String, Vector{Float64}}()
		loglike["early"] = fill(0.0, nruns)
		loglike_nostim["early"] = fill(0.0, nruns)
		loglike_early = fill(0.0, runs)
		loglike_early_late = fill(0.0, runs)

		loglike["mid"] = fill(0.0, nruns) 
		loglike_nostim["mid"] = fill(0.0, nruns) 
		midx = argmin(model_bic["nostim"])
		m = n_modes["nostim"][midx]
		ntot = length(rtime_nostim)
		for (ll,x) in zip(["early","mid"],[rtime_early, rtime_mid])
			ntest = length(x)
			ntrain = ntot - ntest
			for r in 1:nruns 
				# pick a random subset of the non-stimulated trials for training
				tridx = shuffle(1:ntot)[1:ntrain]
				sort!(tridx)
				teidx = setdiff(1:ntot, tridx)
				if m == 1
					gm = fit(Gamma, rtime_nostim[tridx])
				else
					gm,_ = fit(GammaMixture, rtime_nostim[tridx],m;niter=20_000)
				end
				# test both on remaining stim data and on th stim early
				loglike_nostim[ll][r] = loglikelihood(gm, rtime_nostim[teidx])
				loglike[ll][r] = loglikelihood(gm, x)
			end
		end
		midx = argmin(model_bic["early"])
		m = n_modes["early"]
		# bootstrap
		_loglike1 = fill(0.0, nruns)
		_loglike2 = fill(0.0, nruns)
		for r in 1:nruns
			# train on half, test on half
			n = length(rttime_early)
			ntrain = div(n,2)
			tridx = shuffle(1:n)[1:ntrain]
			testidx = setdiff(1:n, tridx)
			gm,_ = fit(GammaMixture, rtime_nostim[tridx],m;niter=20_000)
			_loglike1[r] = loglikelihood(gm, rtime_early[testidx]) 
			testidx2 = shuffle(1:length(rtime_mid))[1:length(testidx)]
			_loglike2[r] = loglikelihood(gm, rtime_mid[testidx2]) 
		end
		JLD2.save(fname, Dict("model_bic"=>model_bic, "n_modes"=>n_modes,"mode_assignment"=>mode_assignment,
						     "model_converged"=>model_converged,"model"=>model,
							 "loglike"=>loglike,"loglike_nostim"=>loglike_nostim,
							 "loglike_early_early"=>_loglike1, "loglike_early_mid"=>_loglike2))
	end

	#limits
	ymin,ymax = extrema([rtime_nostim;rtime_early;rtime_mid])
	Δy = ymax-ymin 
	ymin = ymin - 0.05*Δy
	ymax = ymax + 0.05*Δy
	# create plots of BIC for each model, for early and mid stimulation
	colors = [to_color(:gray); Makie.wong_colors()[3:4]]
	with_theme(plot_theme) do
		fig = Figure()
		#axes = [Axis(fig[1,i]) for i in 1:3]
		labels = [Label(fig[1,i], ll;tellwidth=false) for (i,ll) in enumerate(["No stim","early stim", "late stim"])]
		lg1 = [GridLayout(fig[2,i]) for i in 1:3]
		for (ll,lg,cc) in zip(["nostim","early","mid"], lg1,colors[1:3])
			ax = Axis(lg[1,1])
			barplot!(ax, [1:2;], model_bic[ll][1:2].-model_bic[ll][3],color=cc)
			best_bic, best_model_idx = findmin(model_bic[ll])
			ax.xticks = ([1:2;],string.([3,2]))
			ax.xlabel = "# modes"
			if plot_loglikelihood
				ax2 = Axis(lg[2,1])
				if ll in ["early","mid"]
					# also indicate the loglikehlihood
					colsize!(lg, 1, Relative(0.8))
					lower, mm,upper = percentile(loglike[ll],[5,50,95])
					lower_ns, mm_ns,upper_ns = percentile(loglike_nostim[ll],[5,50,95])
					barplot!(ax2, [1.0,2.0], [mm_ns, mm], color=[colors[1], cc])
					rangebars!(ax2, [1.0, 2.0], [lower_ns, lower], [upper_ns, upper],color=:black)
					ax2.ylabel = "Log-likelihood" 
					ax2.xticksvisible = false
					ax2.xticklabelsvisible = false
				end
			end
			ax.ylabel = "ΔBIC"
		end

		lg2 = [GridLayout(fig[3,i]) for i in 1:3]
		for (ll,x, lg,cc) in zip(["nostim","early","mid"], [rtime_nostim, rtime_early, rtime_mid],lg2,colors)
			# make one color a little brighter, one a little darker
			q = 0.7
			p = 1.0 -q
			cc0 = RGB(q*cc.r, q*cc.g, q*cc.b)
			cc1 = RGB(p + q*cc.r, p+q*cc.g, p+q*cc.b)
			_colors = [cc0, cc, cc1]
			ax1 = Axis(lg[1,1])
			ax2 = Axis(lg[1,2])
			#linkyaxes!(ax1, ax2)
			best_model = model[ll]
			xs = sort(x)
			probs = component_pdf(best_model, xs)
			for c in 1:size(probs,2)
				lines!(ax2, probs[:,c], xs,color=_colors[c])
			end
			scatter!(ax1, rand(length(x)), x,color=_colors[mode_assignment[ll]],markersize=7.5px)
			ylims!(ax1, ymin, ymax)
			ylims!(ax2, ymin, ymax)
			colsize!(lg, 1, Relative(0.7))
			ax2.yticksvisible = false
			ax2.yticklabelsvisible = false
			ax2.xticklabelsvisible = false
			ax2.xlabel = "pdf"
			ax1.xticksvisible = false
			ax1.xticklabelsvisible = false
			if ll == "nostim"
				ax1.ylabel = "Reaction time [ms]"
			end
		end
		# tweak the row size
		rowsize!(fig.layout, 2, Relative(0.3))
		fig
	end
end
end