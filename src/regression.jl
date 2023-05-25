using MultivariateStats
using Distributions
using LinearAlgebra
using StatsBase
using Random

include("utils.jl")

"""
```julia
function llsq_stats(X,y;kvs...debug)

Least square regression using `MultivariateStats.llsq`, also returning r² and p-value of the fit, computed via F-test
```
"""
function llsq_stats(X::Matrix{T},y::Vector{T};kvs...) where T <: Real
	β = llsq(X, y)
	p1 = length(β)
	n = length(y)
	prt = X*β[1:end-1] .+ β[end]
	rsst = sum(abs2, y .- mean(y))
	rss1 = sum(abs2, y .- prt)
	F = (rsst - rss1)/(p1-1)
	F /= rss1/(n-p1)
	pv = 1.0 - cdf(FDist(p1-1, n-p1), F)
	r² = 1.0 - rss1/rsst
	β, r², pv, rss1
end

adjusted_r²(r²::Float64, n::Int64, p::Real) = 1.0 - (1.0-r²)*(n-1)/(n-p)

function ftest(rss1,p1, rss2,p2,n)
    if rss1 > rss2
        fv = (rss1-rss2)/(p2-p1)
        fv /= rss2/(n-p2)
        dp = p2-p1
        p = p2
    else
        fv = (rss2-rss1)/(p1-p2)
        fv /= rss1/(n-p1)
        dp = p1-p2
        p = p1
    end
    fv, 1-cdf(FDist(dp,n-p), fv)
end

"""
````
function regress_reaction_time(;rtmin=120.0, rtmax=300.0, window=35.0, align=:mov, do_shuffle=false, nruns=100)
````

Regress reaction time as a function of time using population responses per session and location
"""
function regress_reaction_time(;rtmin=120.0, rtmax=300.0, window=35.0, align=:mov, do_shuffle=false, nruns=100)
    
    fname = joinpath("data","ppsth_fef_$(align).jld2" )
    ppsth,tlabels,trialidx, rtimes = JLD2.load(fname, "ppsth","labels","trialidx","rtimes")

	# Per session, per target regression, combine across to compute rv
	all_sessions = DPHT.get_level_path.("session", ppsth.cellnames)
	sessions = unique(all_sessions)
	rtp = fill(0.0, 0, size(ppsth.counts,1))
	rt = fill(0.0, 0)
	bins2 = ppsth.bins
	rtime_all = Float64[]
    r2j = fill(0.0, size(ppsth.counts,1),100)
    total_idx = 1
	for (ii, session) in enumerate(sessions)
		X, _label, _rtime = get_session_data(ii,ppsth, trialidx, tlabels, rtimes;rtime_min=rtmin,rtime_max=rtmax)
		X2,bins2 = rebin2(X, ppsth.bins, window)
		ulabel = unique(_label)

		append!(rtime_all, _rtime)
		_rtp = fill(0.0, length(_label), size(X2,1))
		_rt = log.(_rtime)
		for l in ulabel
			tidx = _label.==l
			_rt[tidx] .-= mean(_rt[tidx])
            rsst = sum(abs2, _rt[tidx])
            n = sum(tidx)
			if do_shuffle
				_rt[tidx] .= shuffle(_rt[tidx])
			end
			Xq = cat(permutedims(X2[:,tidx,:],[2,3,1]), fill(1.0, sum(tidx),1, size(X2,1)),dims=2)
			for i in 1:size(Xq,3)
				Xqi = Xq[:,:,i]
				β = ridge(Xqi, _rt[tidx],0.05;bias=false)
				_rtp[tidx,i] .= Xqi*β
                rss = sum(abs2, _rt[tidx] - _rtp[tidx,i])
                _r² = 1.0 - rss/rsst
                # get the degrees of freedom
                p0 = length(β)
                p = tr(Xqi*inv(Xqi'*Xqi + 0.05I)*Xqi')
                r2j[i, total_idx] = adjusted_r²(_r²,n,p)
			end
            total_idx += 1
		end
		rtp = cat(rtp, _rtp, dims=1)
		rt = cat(rt, _rt, dims=1)
	end
	r2 = fill(0.0, size(ppsth.counts,1),nruns)
	for i in 1:nruns
		tidx = rand(1:length(rt), length(rt))
		v1 = sum(abs2, rt[tidx] .- mean(rt[tidx]))
		v2 = sum(abs2, rt[tidx] .- rtp[tidx,:], dims=1)
		r2[:,i] = 1.0 .- v2./v1
	end
	r2, bins2, percentile(rtime_all, [5,95]), r2j[:,1:total_idx]
end

function explain_rtime_variance(subject::String,alignment::Symbol;reg_window=(-400.0, -50.0), rtmin=120.0, rtmax=300.0, realign=false, area="FEF", kvs...)
   fname = joinpath("data","ppsth_$(alignment)_new.jld2")
   ppstht, tlabelst, rtimest, trialidxt = JLD2.load(fname, "ppsth","labels", "rtimes", "trialidx")
   # get all cells from the specified subject
   subject_index = findall(cell->DPHT.get_level_name("subject", cell)==subject, ppstht.cellnames)
   # get the cells for this subject from the specified area
   cellidx = get_area_index(ppstht.cellnames[subject_index], area)
   # index into the full array of cells
   cellidx = subject_index[cellidx]
   all_sessions = DPHT.get_level_path.("session", ppstht.cellnames[cellidx])
   sessions = unique(all_sessions)
   bins = ppstht.bins
   lrt = Float64[]
   Yp = Float64[]
   qridx = Int64[]
   offset = 0
   for (sessionidx,session) in enumerate(sessions)
       X, labels, rtimes = get_session_data(session, ppstht, trialidxt, tlabelst, rtimest, cellidx;rtime_min=rtmin, rtime_max=rtmax,
                                                                                                   variance_stabilize=true,
                                                                                                    mean_subtract=true)
       bidx = searchsortedfirst(bins, reg_window[1]):searchsortedlast(bins, reg_window[2])
       for location in locations[subject]
           tidx = labels.==location
           Xl = dropdims(mean(X[bidx,tidx,:],dims=1),dims=1)
           Σ = cov(Xl, dims=1)
           σ² = diag(Σ)
           try
              # exclude cells with very low variance
               pp = StatsBase.fit(Gamma, σ²)
               ppq = quantile(pp, 0.01)
               cidx = findall(σ² .> ppq)
               cidx = 1:length(σ²)
               Xlc = permutedims(Xl[:, cidx],[2,1])
               @debug "Cells" olength=size(Xl,2) klength=length(cidx)
               fa = StatsBase.fit(MultivariateStats.FactorAnalysis, Xlc;maxoutdim=1, method=:em)
               Y = permutedims(MultivariateStats.predict(fa, Xlc), [2,1])
               Y .-= mean(Y)
               # mirror
               _lrt = log.(rtimes[tidx])
               _lrt .-= mean(_lrt)
               if realign
                   # check the sign of the correlation between reaction time and activity and realign so that the relationship is positive
                   β,r², pv,rss = llsq_stats(Y, _lrt)
                   if β[1] < 0.0
                       Y .*= -1.0
                   end
               end
               append!(lrt, _lrt)
               append!(Yp, Y[:])
               append!(qridx, offset .+ [1:length(_lrt);])
            catch ee
               if isa(ee, DomainError)
                   # failure in fit, continue to next location
                   continue
               else
                   rethrow(ee)
               end
            finally
                offset += sum(tidx)
           end
       end
   end
   lrt, Yp, qridx
end