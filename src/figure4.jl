module Figure4
using AttractorModels
using ProgressMeter
using Makie: Point2f, Point3f
using GLMakie
using CRC32c
using JLD2
using StableRNGs
using KernelFunctions
using Colors
using Loess
using ..Utils
using ..PlotUtils


#include("utils.jl")
include("trajectories.jl")
include("regression.jl")
#include("plot_utils.jl")

get_functions(;kvs...) = AttractorModels.get_attractors2(;w1=sqrt(10.0/2), w2=sqrt(45.0/2.0), wf=sqrt(5.0/2),
                                                        b=-4.5, ϵ2=2.5, A0=7.0, A2=7.0, zmin=-3.5,
                                                        ϕ=3π/4,kvs...)

function generate_trajectories(;do_record=true)
    GLMakie.activate!()
    func,gfunc,ifunc = get_functions()
    curve_data_file = joinpath("data","model_output_more_trials_longer.jld2")
    fig = AttractorModels.animate_manifold(func, gfunc, ifunc;bump_dur=3, nframes=300, bump_amp=0.0,
                                                              σn=0.0525, dt=0.5, max_width_scale=1.0,
                                                              rebound=false, ntrials=500,
                                                              freeze_before_bump=false, r0=1.0,
                                                              b20=7.0, well_min=7.0, basin_scale_min=1.0,
                                                              bump_time=50, do_save=true, zmin_f=0.001,
                                                              zf0=3.5, b0=4.5, ϵ0=2.5, ϵf=1.0,
                                                              do_record=do_record,animation_filename="test_movie.mp4",
                                                              fps=60.0,
                                                              fname=curve_data_file)
end

"""
````
function remove_dependence(Z0, pl, rt)
````
Remove trials until Z0 no longer adds information above that which is contained in pl about `rt`
"""
function remove_dependence(Z0::Vector{Float64}, path_length::Vector{Float64}, rt,α=0.05)
    nt = length(rt)
    βpl, r², pv, rsspl = llsq_stats(repeat(path_length,1,1), rt)
	rtpl = βpl[1].*path_length .+ βpl[end]
	βhr, r², pv, rsshr = llsq_stats([Z0 path_length], rt)
	rthr = [Z0 path_length]*βhr[1:2] .+ βhr[end]
    fv, pv = ftest(rsspl, length(βpl), rsshr, length(βhr), length(rt))
    Δrt = rtpl-rthr
	sidx = sortperm(abs.(Δrt))
    qq = 1
    gidx = 1:nt
    while pv < α && nt-qq > 50 
        gidx = setdiff(gidx, sidx[end-qq:end])
        βpl, r², pv, rsspl = llsq_stats(repeat(path_length[gidx],1,1), rt[gidx])
        βhr, r², pv, rsshr = llsq_stats([Z0[gidx] path_length[gidx]], rt[gidx])
        fv, pv = ftest(rsspl, length(βpl), rsshr, length(βhr), length(gidx))
        qq += 1
    end
    if pv < α
        error("P-value remains below $α after removing $qq trials")
    end
    gidx, pv
end

"""
```julia
function run_model(;redo=false, do_save=true,σ²0=1.0,τ=3.0,σ²n=0.0, nd=14,n_init_points=1,
                                               curve_data_file="curved_new_long_data.jld2",
                                               idx0=30,nruns=50)
````
Compute regression statistics on 2D trajectories projected onto an `nd` dimensional space. 

Returns a NamedTuple with the following fields:

`rt` : reaction time of the underyling simulated ata
`r²` : r² for a regression using the full population activity to explain reaction time at each point in time.
`r²0`: r² for regression using the initial position of the trajectories, projected onto a 1D Factor analysis space
       to explain reaction time variance.
`r²pl`: r² for regression using the reduced path length of the trajectories to explain reaction time variance.
"""
function run_model(;redo=false, do_save=true,σ²0=1.0,τ=3.0,σ²n=0.0, nd=[14],n_init_points=1,
                                               curve_data_file="model_output_more_trials_longer.jld2",
                                               idx0=30,nruns=50, ntrials::Union{Int64, Vector{Int64}}=0,rseed=UInt32(1234),
                                               go_cue=idx0, path_length_method::Symbol=:normal,
                                               remove_outliers=false, do_interpolation=true, do_remove_dependence=true, h0=UInt32(0))
    @assert σ²0 >= σ²n
    h = h0
    h = crc32c(string(σ²0),h)
    h = crc32c(string(τ),h)
    h = crc32c(string(σ²n),h)
    if nd != 7
        if length(nd) == 1
            h = crc32c(string(first(nd)),h)
        else
            h = crc32c(string(nd),h)
        end
    end
    if n_init_points != 1
        h = crc32c(string(n_init_points),h)
    end
    if curve_data_file != "curved_new_long_data.jld2"
        h = crc32c(curve_data_file,h)
    end
    if idx0 != 30
        h = crc32c(string(idx0),h)
    end
    if go_cue != 30
        h = crc32c(string(go_cue),h)
    end
    if nruns != 50
        h = crc32c(string(nruns),h)
    end
    if ntrials != 0 
        h = crc32c(string(ntrials),h)
    end
    if rseed != UInt32(1234)
        h = crc32c(string(rseed),h)
    end
    if path_length_method != :normal
        h = crc32c(string(path_length_method),h)
    end
    if remove_outliers
        h = crc32c("remove_outliers",h)
    end
    if do_interpolation == false
        h = crc32c("no_interpolation",h)
    end
    if !do_remove_dependence
        h = crc32c("keep_dependence",h)
    end
    q = string(h, base=16)
    fname = joinpath("data","model_full_space_results_$q.jld2")
    @show fname
    if isfile(fname) && !redo
        qq = JLD2.load(fname)
        results = NamedTuple(Symbol.(keys(qq)) .=> values(qq))
    else
        xy2 = [7.0 -13.0] # pre-mov basin
        w2 = 35.0 # pre-mov basin width
        curvex, curvey = JLD2.load(curve_data_file,"curvex","curvey")
        # split into trials
        pidx = findall(isnan, curvex)
        curves = [[curvex[pp0+1:pp1-1] curvey[pp0+1:pp1-1]] for (pp0,pp1) in zip(pidx[1:end-1], pidx[2:end])]
        # compute reaction time
        eeidx = fill(0, length(curves))
        eeidxs = fill!(similar(eeidx), 0)
        curvesp = Vector{Matrix{Float64}}(undef, length(curves))
        path_length = fill(0.0, length(curves))
        path_length_tr = fill(0.0, length(curves))
        offset = go_cue - idx0
        if do_interpolation
            offset *= 10
        end
        for (ii,curve) in enumerate(curves)
            # to get a finer resolution reaction time, we'll first upsample the trajectories and repeat the above analysis
            if do_interpolation
                spl = ParametricSpline(permutedims(curve[idx0:end,:], [2,1]))
                curvesp[ii] = permutedims(Dierckx.evaluate(spl, range(extrema(spl.t)...; length=10*length(spl.t))),[2,1])
            else
                curvesp[ii] = curve
            end
            d = sqrt.(dropdims(sum(abs2, curvesp[ii] .- xy2,dims=2),dims=2))
            _eeidx = findfirst(d .< 0.01*w2)
            if _eeidx != nothing
                eeidx[ii] = _eeidx
                path_length[ii] = sum(sqrt.(sum(abs2, diff(curvesp[ii][offset+1:_eeidx,:],dims=1),dims=2)))
                path_length_tr[ii],_ = compute_triangular_path_length(curvesp[ii][offset+1:_eeidx,:], path_length_method)
            end
        end
        _Y0 = cat([curve[offset+1,:] for curve in curvesp]..., dims=2)
        fa = MultivariateStats.fit(MultivariateStats.FactorAnalysis, _Y0;maxoutdim=1,method=:em)
        _Z0 = MultivariateStats.predict(fa, _Y0)
        # we want to make sure that in the underlying data, all information about reaction time is already contained in the path length.
        # to this end, we remove trials for which that might not be true
        tidx = findall(path_length .> 0.0) 
        rt = eeidx[tidx] .- offset .+ 1
        if remove_outliers
            _tidx = findall(rt .<= percentile(rt, 90))
            rt = rt[_tidx]
            tidx = tidx[_tidx]
        end
        # rescale to get the same ballback reaction times as in the data; this is just for show
        rt  = 150.0(rt .- minimum(rt))./(maximum(rt)-minimum(rt)) .+ 120.0

        if do_remove_dependence
            gidx, pv = remove_dependence(_Z0[1,tidx],path_length[tidx], rt)
            @info "Sanitize pv" pv length(gidx)
        else
            gidx = 1:length(tidx)
        end
        rt = rt[gidx]
        tidx = tidx[gidx]
        path_length = path_length[tidx]
        path_length_tr = path_length_tr[tidx]
        @info "tidx" length(tidx)
        #rt = 10*path_length
        #ntrials = length(tidx)
        @info "Rt-range" extrema(rt)
        eeidx = eeidx[tidx]
        curvesp = curvesp[tidx]
        max_rt_idx = argmax(rt)
        min_rt_idx = argmin(rt)
        np_min = 10
        # 
        Δt = round(Int64, (eeidx[min_rt_idx]-1)/np_min)
        Δt = 1
        if ntrials != 0
            n_tot_trials = sum(ntrials)
        else
            n_tot_trials = length(curvesp)
        end
        # create a cue-aligned higher dimensional population response
        ridxf = 0
        Z0 = fill(0.0, n_tot_trials,nruns)
        pl = fill(0.0, length(curvesp))
        σ² = fill(0.0, size(curvesp[1],1))
        for i in 1:length(σ²)
            _Σ = cov(cat([curve[i,:] for curve in curvesp]...,dims=2),dims=2)
            u,s,v = svd(_Σ)
            σ²[i] = sum(s) 
        end
        pltr = fill(0.0, n_tot_trials, nruns)
        plftr = fill(0.0, n_tot_trials, nruns)
        asftr = fill(0.0, n_tot_trials, nruns)
        plf = fill(0.0, n_tot_trials, nruns)
        pl = fill(0.0, n_tot_trials)
        r²0 = fill(0.0, nruns)
        pv0 = fill(0.0, nruns)
        r²m = fill(0.0, nruns)
        r²e = fill(0.0, nruns)
        r²pl = fill(0.0, nruns)
        pvpl = fill(0.0, nruns)
        r²plf = fill(0.0, nruns)
        r²asftr = fill(0.0, nruns)
        r²pcas = fill(0.0, nruns)
        r²hr = fill(0.0, nruns)
        r²cv = fill(0.0, nruns)
        pvf = fill(0.0, nruns)
        fvalue = fill(0.0, nruns)
        # per session regressions
        r² = fill(0.0, size(curvesp[1],1), length(nd), nruns)
        r²s = fill!(similar(r²), 0.0)
        σ²f = fill!(similar(r²), 0.0)
        rt_tot = fill(0.0, n_tot_trials,nruns)
        # Sample data for one session
        _nt = ntrials != 0 ? first(ntrials) : length(curvesp)
        Yf = fill(0.0, size(curvesp[1],1), nd[1], _nt)
        Yq = fill!(similar(Yf), 0.0)
        Ysq = fill!(similar(Yf), 0.0)
        Ysf = fill!(similar(Yf), 0.0)
        rtf = fill(0.0, _nt)
        eeidxf = fill(0, _nt)
        eeidxq = fill(0, _nt)
        rtq = fill(0.0, _nt)
        # correlated noise
        A = Vector{Matrix{Float64}}(undef, length(nd))
        # scaled square exponential kernel
        kernel = compose(SqExponentialKernel(), ScaleTransform(1.0/τ))
        tt = 1.0:size(curvesp[1],1)
        for ii in 1:length(A)
            _kernel = WhiteKernel()*ConstantKernel(;c=σ²n/nd[ii]) + kernel*ConstantKernel(;c=(σ²0-σ²n)/nd[ii])
            K = kernelmatrix(_kernel,tt)
            A[ii] = first(cholesky(K + 1e-6*I))
        end
        #Q = [correlated_noise_process(1.0:size(curvesp[1],1), τ,1, (σ²0-σ²n)/nd[kk], σ²n/nd[kk]) for kk in 1:length(nd)]
        #A = [first(cholesky(q))  for q in Q]
        r²0min = Inf
        RNG = StableRNG(rseed)
        @showprogress for r in 1:nruns
            # maybe we need to do some smoothing here?
            trial_offset = 0
            for (kk,(_ntrials, _ncells)) in enumerate(zip(ntrials,nd))
                if _ntrials != 0
                    if _ntrials < length(curvesp)
                        # sub-sample
                        tidx = shuffle(1:length(curvesp))[1:_ntrials]
                        sort!(tidx)
                        _curvesp = curvesp[tidx]
                        _eeidx = eeidx[tidx]
                        _rt = rt[tidx]
                        _path_length = path_length[tidx]
                    else
                        _curvesp = curvesp
                        _eeidx = eeidx
                        _rt = rt
                        _ntrials = length(_rt)
                        _path_length = path_length
                    end
                else
                    _curvesp = curvesp
                    _eeidx = eeidx
                    _rt = rt
                    _ntrials = length(_rt)
                    _path_length = path_length
                end
                qq,_ = qr(randn(_ncells,_ncells))
                W = qq[:,1:2]
                Y = fill(0.0, size(_curvesp[1],1),_ncells,_ntrials)
                Ys = fill!(similar(Y), NaN)
                for i in 1:_ntrials
                    curve = _curvesp[i]
                    for j in axes(curve,1)
                        Y[j,:,i] .= W*curve[j,:] 
                    end
                    # add correlated noise
                    Y[:,:,i] .+= A[kk]*randn(RNG, size(A[kk],1),_ncells)
                    plf[trial_offset+i,r] = sum(sqrt.(sum(abs2, diff(Y[offset+1:_eeidx[i],:,i],dims=1),dims=2)))
                    plftr[trial_offset+i,r],pidx = compute_triangular_path_length(Y[offset+1:_eeidx[i],:,i],path_length_method)
                    # compute speed
                    asftr[trial_offset+i, r] = plf[trial_offset+i,r]/(_eeidx[i]-offset+2)
                end
                min_rt_idx = argmin(_rt)
                Δ = plf[trial_offset + min_rt_idx,r]/np_min
                if kk == 1
                    Yq .= Y
                    Ysq .= Ys
                    rtq .= _rt
                    eeidxq .= _eeidx
                end
                rt_tot[trial_offset+1:trial_offset+_ntrials,r] .= _rt
                # initial position
                # To make it as similar to the model as possible, we use factor analysis here
                Y0 = dropdims(mean(Y[offset+1:offset+n_init_points,:,:],dims=1),dims=1)
                fa = MultivariateStats.fit(MultivariateStats.FactorAnalysis, Y0;maxoutdim=1,method=:em)
                Z0[trial_offset+1:trial_offset+_ntrials,r] = permutedims(MultivariateStats.predict(fa, Y0),[2,1])
                _rtl = log.(_rt)
                _rts = shuffle(RNG, _rtl)
                for i in 1:length(σ²)
                    _Σ = cov(Y[i,:,:],dims=2)
                    u,s,v = svd(_Σ)
                    σ²f[i,kk,r] = sum(s)
                    β, r²[i,kk,r], _, _ = llsq_stats(permutedims(Y[i,:,:],[2,1]),_rtl) 
                    β, r²s[i,kk,r], _, _ = llsq_stats(permutedims(Y[i,:,:],[2,1]), _rts)
                end
                trial_offset += _ntrials
            end
            rtl = log.(rt_tot[:,r])
            β0,r²0[r], pv0[r],rss0 = llsq_stats(Z0[:,r:r], rtl)
            @debug "Regress initial" r²0[r] β
            βpl,r²pl[r], pvpl[r],rsspl = llsq_stats(plftr[:,r:r], rtl)
            @debug "Regress path length" r²pl[r]
            if pv0[r] < 0.01 && pvpl[r] < 0.01 && r²0[r] < r²0min
                Yf .= Yq
                Ysf .= Ysq
                rtf .= rtq
                eeidxf .= eeidxq
                r²0min = r²0[r]
                ridxf = r
            end
            βplf,r²plf[r], pvplf,rssplf = llsq_stats(plf[:,r:r], rtl)
            βasf, r²asftr[r], _,_ = llsq_stats(asftr[:,r:r], rtl)
            #r²plf[r] = adjusted_r²(r²plf[r], ntrials, length(βplf))
            @debug "Regress path length" r²plf[r]
            # hierarhical
            βhr,r²hr[r], pvhr,rsshr = llsq_stats([Z0[:,r] plftr[:,r]], rtl)
            βpcas, r²pcas[r],_,_ = llsq_stats([Z0[:,r] asftr[:,r]], rtl)
            @debug "ZO" extrema(Z0[:,r]) extrema(plftr[:,r]) rsshr rsspl
            fvalue[r],pvf[r] = ftest(rsspl,length(βpl), rsshr, length(βhr),length(rtl)) 
            #r²hr[r] = adjusted_r²(r²hr[r], ntrials, length(βhr))
            @debug "Regress path length + initial" r²hr[r]
            # compute variance as function of time
            #cross-validated to make sure that the path lengths do carry information
            rtl = log.(rt)
            tidx = shuffle(1:length(curvesp))[1:div(length(curvesp),2)]
            teidx = setdiff(1:length(curvesp),tidx)
            βcv,r²cv[r], pvcv,rsscv = llsq_stats(repeat(path_length[tidx],1,1), rtl[tidx])
            rtp = βcv[1].*path_length[teidx] .+ βcv[end]
            r²cv[r] = 1.0 - sum(abs2, rtp - rtl[teidx])/sum(abs2, rtl[teidx] .- mean(rtl[teidx]))
            r²cv[r] = adjusted_r²(r²cv[r], length(curvesp), length(βcv))
            
        end
        results = (r²0=r²0, pv0=pv0, r²pl=r²pl, pvpl=pvpl, r²hr=r²hr, r²=r², r²s=r²s,eeidx=eeidx,
                   rt=rt_tot, rt_orig=rt, pltr=pltr, σ²f=σ²f, curves=curvesp,σ²0=σ²0, σ²=σ²,Z0=Z0, Y=Yf, Ys=Ysf, rt_sample=rtf, eeidx_sample=eeidxf, runidx=ridxf,
                   path_length=path_length, path_length_tr=path_length_tr, plf=plf, plftr=plftr, r²plf=r²plf, r²asftr=r²asftr, r²pcas=r²pcas, r²cv=r²cv, fvalue_init_pl=fvalue, pvalue_init_pl=pvf)
        @info "r²plf" r²plf
        if do_save
            JLD2.save(fname, Dict(String(k)=>results[k] for k in keys(results)))
        end
    end
    results
end

function plot(;redo=false, width=700,height=700, do_save=true,h0=one(UInt32), kvs...)
    RNG = StableRNG(UInt32(1234))
	Xe = [7.0, -13.0]
	w2 = 35.0
	do_interpolation = false
	results = run_model(;redo=redo,σ²0=1.125,τ=1.0,σ²n=0.0,nd=ncells["W"],
                                          n_init_points=1, curve_data_file="model_output_more_trials_longer.jld2",
                                          idx0=1,go_cue=50, nruns=50,ntrials=_ntrials["W"],
                                          path_length_method=:normal, remove_outliers=true,
                                          do_interpolation=false, do_remove_dependence=true, do_save=do_save,h0=h0, kvs...);
	# the function that was used to generate the trajectories
	func,gfunc,ifunc = get_functions() 

	xx = range(-10.0, stop=20.0, length=200);
	yy = range(-25.0, stop=5.0, length=200)

	#sub-sample to 50 curves
	#tidx = sort(shuffle(RNG, 1:length(curves))[1:50])

	if do_interpolation
		idx0 = 50 
	else
		idx0 = 50
	end
	# compute path length and initial conditions
	path_length = fill(0.0, size(results.Y,3))
    avg_speed = fill!(similar(path_length), 0.0)
	for i in 1:size(results.Y,3)
		y = results.Y[idx0+1:results.eeidx_sample[i],:,i]
		path_length[i],dm = compute_triangular_path_length2(y)
        avg_speed[i] = sum(sqrt.(sum(abs2,diff(y,dims=1),dims=2)))/(results.eeidx_sample[i]-idx0+2)
	end
	fa = MultivariateStats.fit(MultivariateStats.FactorAnalysis, results.Y[idx0,:,:];method=:em, maxoutdim=1)
	z0 = MultivariateStats.predict(fa, results.Y[idx0,:,:])

	cm = to_colormap(:berlin)

	# get curves for plotting
	# TODO: Plot fewer paths
	# plot 10 trials spanning the reaction time distribution
	qidx = sortperm(results.path_length)
	flat_curves_x = Float64[] 
	flat_curves_y = Float64[]
	ttidx = qidx[[1,div(length(qidx),2),length(qidx)]]
	tvidx = qidx[round.(Int64, range(1,stop=length(qidx), length=10))]
	icidx = Bool[]
	ucidx = Bool[]
	flat_colors = eltype(cm)[]
	for (ii,curve) in enumerate(results.curves)
		append!(flat_curves_x, curve[:,1])
		push!(flat_curves_x, NaN)
		append!(flat_curves_y, curve[:,2])
		push!(flat_curves_y, NaN)
		if ii in ttidx
			jj = findfirst(ttidx.==ii)
			append!(icidx, fill(true, size(curve,1)+1))
			append!(flat_colors, fill(cm[round(Int64,(jj-1)/(length(ttidx)-1)*(length(cm)-1)+1)],size(curve,1)+1))
		else
			append!(icidx, fill(false, size(curve,1)+1))
		end
		if ii in tvidx
			jj = findfirst(tvidx.==ii)
			append!(ucidx, fill(true, size(curve,1)+1))
		else
			append!(ucidx, fill(false, size(curve,1)+1))
		end
	end

	sidx = sortperm(results.eeidx_sample)
	vidx = invperm(sidx)
	colors = cm[round.(Int64, range(1, stop=length(cm), length=length(sidx)))]
	# sort by reaction time
	# unsort the colors
	colors = colors[vidx]

	single_cell_responses = Vector{Vector{Point2f}}(undef, 3)
	response_colors = Vector{Vector{eltype(colors)}}(undef, 3)
	for (jj,cidx) in enumerate([1,7,15])
		single_cell_responses[jj] = Point2f[]
		response_colors[jj] = eltype(colors)[]
		for i in 1:size(results.Y,3)
			y = results.Y[idx0-15:results.eeidx_sample[i],cidx,i]
			x = [-results.eeidx_sample[i]+idx0-15:0;]
			# smooth out the noise
			model = loess(x, y, span=0.5)
			us = range(extrema(x)...; step = 0.1)
			vs = Loess.predict(model, us)
			append!(single_cell_responses[jj], Point2f.(zip(us,vs)))
			push!(single_cell_responses[jj], Point2f(NaN, NaN))
			append!(response_colors[jj], fill(colors[i], length(us)+1))
		end
	end
	# find a trial with long reaction time
	mx = round(Int64, percentile(results.eeidx_sample .- idx0 .+ 1, 97))
	# find all trials where reaction time + go cue is at least as large the reaction time  
	tqidx = findall( mx + idx0 .> results.eeidx_sample .>= mx)
	@info "Remaining trials" length(tqidx)
	xxr = -mx+1:0.0
	r² = fill(0.0, length(xxr))
	r²s = fill(0.0, length(xxr))
	# for the shorter reaction times we'll just extend into the delay 2 period
	# for those trials with shorter reaction time, we need to include an amount before the go-cue onset equivalent to the difference between the reaction time of those trials and the maximum reaction time.
	Ym = fill(0.0, mx, size(results.Y,2), length(tqidx))
	for i in 1:size(Ym,3)
		Δ = mx - (results.eeidx_sample[tqidx[i]]-idx0)
		Ym[:,:,i] = results.Y[idx0-Δ+1:results.eeidx_sample[tqidx[i]],:, tqidx[i]]
	end
	lrt = log.(results.rt_sample[tqidx])
	lrts = shuffle(lrt)
	for i in 1:length(r²)
		yp = permutedims(Ym[i,:,:], [2,1])
		β, r²[i], pv, rss = llsq_stats(yp, lrt)
		β, r²s[i], pv, rss = llsq_stats(yp, lrts)
	end
	#regression
	βpl, r²pl, pvpl, rsspl = llsq_stats(repeat(path_length[tqidx],1,1), lrt)	
	βpc, r²pc, pvpc, rsspc = llsq_stats(permutedims(z0,[2,1])[tqidx,:], lrt)
	βas, r²as, pvas, rssas = llsq_stats(repeat(avg_speed[tqidx],1,1), lrt)	
	@info r²pl r²pc r²as
	μr = fill(0.0,5)
	lr = fill(0.0,5)
	ur = fill(0.0,5)
	for (ii,r²) in enumerate([results.r²0,results.r²pl, results.r²hr,results.r²asftr, results.r²pcas])
		dd = fit(Beta, r²)
		μr[ii] = mean(dd)
		lr[ii] = quantile(dd, 0.05)
		ur[ii] = quantile(dd, 0.95)
	end
	zmin = -10.0
    plot_colors = Makie.wong_colors()
	with_theme(plot_theme) do
		fig = Figure(resolution=(width,height))
		lg1 = GridLayout()
		fig[1,1] = lg1
		ax1 = Axis3(lg1[1,1],azimuth=5.000607537633862, elevation=0.5089042208588803)
		ax1.xticklabelsize = 12
		ax1.yticklabelsize = 12
		ax1.zticklabelsize = 12
		ax1.zlabelvisible = false
		surface!(ax1, xx, yy, -zmin .+ func.([[x,y] for x in xx, y in yy]), colormap=colormap("rdbu",mid=0.72))
		ax1.xlabel = "Dim 1"
		ax1.ylabel = "Dim 2"
		# add the original paths
		lines!(ax1, flat_curves_x[ucidx], flat_curves_y[ucidx], -zmin .+ func.([[x,y] for (x,y) in zip(flat_curves_x[ucidx], flat_curves_y[ucidx])]), color="black")

		# indicate the limit of the attractor
		lpoints = decompose(Point3f, Circle(Point2f(Xe), 0.01*w2))
		lpoints .+= Point3f(0.0, 0.0, zmin)	
		lines!(ax1, flat_curves_x[icidx], flat_curves_y[icidx], fill(0.0,sum(icidx)),color=flat_colors)
		lines!(ax1, lpoints, color=plot_colors[1])
        #TODO: Plot this on the ``floor``
        contour!(ax1, xx,yy, func.([[x,y] for x in xx, y in yy]),levels=15, colormap=colormap("rdbu",mid=0.72))
		# add panels) showing single unit responses aligned to movement onset
        zlims!(ax1, 0.0, 10.0)
		lg2 = GridLayout()
		fig[1,2] = lg2
		ax2 = Axis(lg2[1,1])
		ax2.xticklabelsvisible = false
		lines!(ax2, single_cell_responses[1], color=response_colors[1])
		ax3 = Axis(lg2[2,1])
		ax3.xticklabelsvisible = false
		lines!(ax3, single_cell_responses[2], color=response_colors[2])
		ax4 = Axis(lg2[3,1])
		lines!(ax4, single_cell_responses[3], color=response_colors[3])
		for ax in [ax2,ax3,ax4]
			vlines!(ax, 0.0, color="black")
		end
		linkxaxes!(ax2, ax3, ax4)
		ax4.xlabel = "Time from movement"
		# add panels for initial and path length
		lg3 = GridLayout()
		lg1[2,1] = lg3
		rowsize!(lg1, 1, Relative(0.8))
		ax5 = Axis(lg3[1,1])
		scatter!(ax5, z0[:], log.(results.rt_sample),color=colors, markersize=7.5px)
		ablines!(ax5, βpc[end], βpc[1], color="black", linestyle=:dot)
		ax5.xlabel = "Initial (MP)"
		ax5.ylabel = "log(rt)"
		ax6 = Axis(lg3[1,3],xticks=LinearTicks(4))
		scatter!(ax6, path_length, log.(results.rt_sample),color=colors, markersize=7.5px)
		ablines!(ax6, βpl[end], βpl[1], color="black", linestyle=:dot)
		ax6.xlabel = "Path length (PL)"
		ax6.yticklabelsvisible = false
        ax73 = Axis(lg3[1,2], xticks=LinearTicks(4))
		scatter!(ax73, avg_speed, log.(results.rt_sample),color=colors, markersize=7.5px)
		ablines!(ax73, βas[end], βas[1], color="black", linestyle=:dot)
		ax73.xlabel = "Avg speed (AS)"
		ax73.yticklabelsvisible = false
        for _ax in [ax5, ax6, ax73]
            _ax.xticklabelsvisible = false
        end
		linkyaxes!(ax5, ax6, ax73)
		#TODO: Add reaction time regression as a function of time
		lg4 = GridLayout()
		fig[2,1:2] = lg4
		ax7 = Axis(lg4[1,1])	
		lines!(ax7, xxr, r², label="Model data")
		lines!(ax7, xxr, r²s, label="Surrogate")
		ax7.xlabel = "Time from movement [au]"
		ax7.xticklabelsvisible = false
		axislegend(ax7,valign=:top, halign=:right, margin=(0.0, -0.0, 0.0, -20.0))
		ax7.ylabel = "r²"
        # reaction time
        ax11 = Axis(lg4[1,3])
        _rtime =results.rt_sample[tqidx] 
        dd = StatsBase.fit(Gamma, _rtime)
        x = sort(_rtime)
        y = pdf.(dd,x)
        y ./= sum(y)
        scatter!(ax11, rand(length(tqidx)),_rtime)
        ax112 = Axis(lg4[1,3])
        linkyaxes!(ax11, ax112)
        lines!(ax112, y, x, color="black")
        ax112.xticklabelsvisible = false 
        ax112.xticksvisible = false 
        ax112.yticklabelsvisible = false
        ax112.yticksvisible = false
        ax112.bottomspinevisible = false
        ax112.leftspinevisible = false
        ax11.xticklabelsvisible = false
        ax11.xticksvisible = false
        ax11.bottomspinevisible = false
        ax11.ylabel = "Reaction time"
		ax8 = Axis(lg4[1,2])
        colsize!(lg4, 1, Relative(0.6))
        colsize!(lg4, 3, Relative(0.2))

		barplot!(ax8, 1:length(ur), μr)
		rangebars!(ax8, 1:length(ur), lr, ur)
		ax8.xticks=([1:length(μr);], ["MP","PL", "MP+PL","AS", "MP+AS"])
        ax8.xticklabelrotation = -π/3
		colgap!(lg4, 1, 30.0)
		label_padding = (0.0, 0.0, 10.0, 1.0)
		labels = [Label(lg1[1,1,TopLeft()], "A",padding=label_padding),
				 Label(lg2[1,1,TopLeft()], "B",padding=label_padding),
				 Label(lg3[1,1,TopLeft()],"C",padding=label_padding),
				 Label(lg4[1,1,TopLeft()], "D",padding=label_padding),
				 Label(lg4[1,2,TopLeft()], "E",padding=label_padding),
                 Label(lg4[1,3,TopLeft()], "F", padding=label_padding)]
		colsize!(fig.layout, 1, Relative(0.7))
		rowsize!(fig.layout, 1, Relative(0.8))
		colgap!(fig.layout, 1, 10.0)
		rowgap!(fig.layout, 1, 1.0)
		fname = joinpath("figures","manuscript","toy_model_figure.png")
        if do_save
            save(fname,fig;pt_per_unit=1)
        end
		fig
	end

end
end #module
