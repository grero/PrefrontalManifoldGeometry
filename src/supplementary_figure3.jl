using EventOnsetDecoding
using LinearAlgebra
using StatsBase
using Random
using JLD2
using HDF5
using ProgressMeter

include("utils.jl")

"""
    get_cross_subspace_decoding(subject::String, train::Symbol, test::Symbol;redo=false)

Get the performance of a decoder trained on one period and tested on another
"""
function get_cross_subspace_decoding(subject::String, train::Symbol, test::Symbol;redo=false,baseline_end=-300.0,nruns=100)
	if subject == "W"
		args =[sessions_w, [1:4;]]
	elseif subject == "J"
		args =[sessions_j, [1:8;]]
	elseif subject == "ALL"
		args = [[sessions_j;sessions_w], [1:8;]]
	end
	outfname = "$(subject)_train_$(train)_test_$(test).jld2"
    window = [range(5.0, step=10.0, stop=50.0);]
    latency = range(100.0, step=-10.0, stop=0.0)
	if isfile(outfname) && !redo
		f1score = JLD2.load(outfname, "f1score")
	else
        ppsth, trialidx, tlabels, rtimes = JLD2.load(joinpath("data","ppsth_$(test).jld2"), "ppsth","trialidx","labels", "rtimes")
		
		fef_idx = findall(get_area_index(ppsth.cellnames, "FEF"))
        push!(args, fef_idx)
		kvs = Pair{Symbol, Any}[]
		#window = range(5.0, step=10.0, stop=50.0)
		#window = [1,2,3,4]
		#remove_window=(4.0, 4.0)
		remove_window = nothing
		reverse_bins = train == :cue
		push!(kvs, :nruns => nruns)
		push!(kvs, :shuffle_bins => false)
		push!(kvs, :use_new_decoder => false)
		push!(kvs, :difference_decoder => true)
		push!(kvs, :windows => window)
		push!(kvs, :latencies => latency)
		push!(kvs, :combine_locations => true)
		push!(kvs, :combine_training_only => false)
		push!(kvs, :shuffle_each_trial => false)
		push!(kvs, :simple_shuffle => false)
		push!(kvs, :restricted_shuffle => false)
		push!(kvs, :shuffle_latency => false)
		push!(kvs, :reverse_bins => reverse_bins)
		push!(kvs ,:fix_resolution => false)
		push!(kvs, :at_source => false)
		push!(kvs, :shuffle_training => false)
		push!(kvs, :remove_window => remove_window)
		push!(kvs, :max_shuffle_latency => 50.0)
		push!(kvs, :rtime_min => 120.0)
		push!(kvs, :mixin_postcue => true)
        push!(kvs, :baseline_end => baseline_end)
        push!(kvs, :save_sample_indices => true)
		dargs = EventOnsetDecoding.DecoderArgs(args...;kvs...)
		fname = joinpath("data", EventOnsetDecoding.get_filename(dargs))
        @show fname
		weights, pmeans, windows, latencies,trainidx = h5open(fname) do fid
			read(fid, "weights"), read(fid, "pmeans"), read(fid, "window"), read(fid,"latency"), read(fid, "training_trial_idx")
		end
        @show dargs.baseline_end
		bins = ppsth.bins
        @show bins
        binsize = bins[2] - bins[1]
		if test == :cue
			bins = -1*reverse(bins)[ppsth.windowsize+1:end]
			bidx = findall(maximum(dargs.windows) .< bins .< -dargs.baseline_end - maximum(dargs.windows))
        else
            bidx = findall(bins[1] + maximum(dargs.windows) .< bins .< dargs.baseline_end)
		end
        @show bins, bidx
		Xtot = fill(0.0, size(ppsth.counts,1), size(ppsth.counts, 2), length(fef_idx))
		label_tot = Vector{Vector{Int64}}(undef, length(fef_idx))
		celloffset = 0
		for i in dargs.sessionidx
			X, _label, _rtime = EventOnsetDecoding.get_session_data(dargs.sessions[i],ppsth, trialidx, tlabels, rtimes, fef_idx;rtime_min=dargs.rtime_min,rtime_max=dargs.rtime_max)
			if test == :cue
				# reverse the bins
				X .= X[end:-1:1,:,:]
			end
			ttidx = findall(in(dargs.locations), _label)
			_label = [findfirst(dargs.locations.==l) for l in _label[ttidx]]
			X = X[:, ttidx,:]
			_rtime = _rtime[ttidx]
			if dargs.combine_locations
				fill!(_label, 1)
			end
			_ncells = size(X,3)
			cidx = celloffset+1:celloffset+_ncells
			Xtot[:, 1:size(X,2),cidx] .= X
			for (jj, cc) in enumerate(cidx)
				label_tot[cc] = _label
			end
			celloffset += _ncells
		end
		Xtot = Xtot[:,:,1:celloffset]
		label_tot = label_tot[1:celloffset]
		ncells = size(Xtot,3)
		ulabel = union(map(unique, label_tot)...)
		use_locations = ulabel
		sort!(use_locations)
		nlocations = length(use_locations)
		RNGs = [MersenneTwister(rand(UInt32)) for r in 1:nruns]
		f1score = fill(0.0, length(windows), length(latencies), length(RNGs))
        ntest = 300
        prog = Progress(nruns*length(windows)*length(latencies))
        
		Threads.@threads for r in 1:length(RNGs)
            # TODO: Make sure to not use the trainidx here.
            Ytest = fill(0.0, size(Xtot,1), ntest, size(Xtot,3))
            testidx = [Int64[] for i in 1:size(Xtot,3)]
            for i in axes(Ytest,3)
                _trainidx = unique(trainidx[i,:,r])
                _testidx = setdiff(1:length(label_tot[i]), _trainidx)
                sort!(_testidx)
                testidx[i] = _testidx
            end
            EventOnsetDecoding.sample_trials!(Ytest, Xtot, label_tot, testidx;RNG=RNGs[r])
			#Yt, train_label,test_label =  EventOnsetDecoding.sample_trials(permutedims(Xtot,[3,2,1]), label_tot;RNG=RNGs[r])
			
			for (iw,w) in enumerate(dargs.windows)
				for (il, l) in enumerate(dargs.latencies)
                    # mayb doesn't work for cue?
                    if test == :cue
                        eidx = findall(-l .<= bins .< -l + w)
                    else
                        eidx = findall(-l - w .<= bins .< -l)
                    end
					midx = first(eidx)
					W = weights[:,1:1,iw,il,1,r]
					pm = pmeans[1:1,:,iw,il,1,r]
					tp = 0.0
					fp = 0.0
					fn = 0.0
					nn = 0
					np = 0
                    wl = round(Int64, w/binsize)
                    midx = searchsortedfirst(bins[1+wl:end], bins[midx])
                    Q = fill(0, size(Ytest,1)-2*wl, size(Ytest,2))
                    # restrict the baseline indices
                    b2idx = findall(in(bidx), wl:(wl+size(Q,1)-1))
                    #@show b2idx, size(Q)
					for tt in 1:size(Ytest,2)
						#project onto space
                        j = midx
                        for j in axes(Q,1)
                            y = sum(Ytest[j+wl:j+2*wl-1,tt,:],dims=1) - sum(Ytest[j:j+wl-1,tt,:],dims=1)
                            #q = (Ytest2[2:end,tt,:] -Ytest2[1:end-1,tt,:])*W
                            # [1xnc] [ncxm]
                            q = permutedims(y*W)
                            # [mx1]
                            if tt == 1 && j == 1
                                #@show size(q) size(pm)
                            end
                            d = dropdims(sum(abs2,q .- pm,dims=1),dims=1)
                            cq = argmin(d)
                            Q[j,tt] = cq
                            #cq = dropdims(argmin(d,dims=2),dims=2)
                        end
						#llb = dropdims(argmin(d,dims=))
						if rand() < 0.5
                            # TODO: Check if this works
							_bidx = rand(b2idx)
							#fp += cq[_bidx].I[2] == 2
                            fp += Q[_bidx,tt] == 2
							nn += 1
						else
							#tp += cq[midx].I[2] .== 2
							#fn += cq[midx].I[2] .== 1
                            tp += Q[midx, tt] .== 2
                            fn += Q[midx, tt] .== 1
							np += 1
						end
					end
					tp /= np
					fn /= np
					fp /= nn
					#find the closest
                    # artificially high f1score could be because either fn or fp are too low.
					f1score[iw,il,r] = tp/(tp + 0.5*(fn + fp))
                    next!(prog; showvalues=[(:run, r)])
				end
			end
		end
		JLD2.save(outfname, Dict("f1score" => f1score, "windows"=>windows, "latencies"=>latencies))
	end
	f1score, train, test, window, latency;
end

function plot(;do_save=true)
    with_theme(plot_theme)  do
        fig = Figure(resolution=(700,400))
        # train on cue, test on mov 
        lg1 = GridLayout()
        fig[1,1] = lg1
        f1score, trainq, testq, windows,latencies = get_cross_subspace_decoding("ALL", :cue, :mov;redo=false,baseline_end=-250.0)
        h1,ax1 = plot_performance!(lg1, f1score, windows, latencies;show_colorbar=false, colormap=:Blues, colorrange=(0.0, 0.72))
        ax1.title = "Trained on go-cue,\ntested on movement"
        ax1.ylabel = "Latency [ms]"
        ax1.xlabel = "Window [ms]"

        # train on mov test on cue 
        lg2 = GridLayout()
        fig[1,2] = lg2
        f1score, trainq, testq, windows,latencies = get_cross_subspace_decoding("ALL", :mov, :cue;redo=false,baseline_end=-300.0)
        h2,ax2 = plot_performance!(lg2, f1score, windows, latencies;show_colorbar=false, colormap=:Blues, colorrange=(0.0, 0.72))
        ax2.title = "Trained on movement,\ntested on go-cue"
        ax2.yticklabelsvisible = false
        Colorbar(fig[1,3], h2, label="F₁ sccore")
        if do_save
            fname = joinpath("figures","manuscript","supplementary_figure4.pdf")
            save(fname, fig;pt_per_unit=1)
        end
        fig
   end
end