"""
```julia
function get_shared_dimensionality(subject::String, sessionidx::Int64,location::Int64; reg_window=(-400.0, -50.0), rtmin=120.0, rtmax=300.0)

Get the number of dimensions needed to explain at least 95% of the shared variance, computed via a 2-step Factor Analysis.

Step 1, find the number of factors what maximises the log-likelihood of the data, via cross-validation.
Step 2, find the number of factors necessary to explain at least 95% of the shared variance for the optimal number of factors found in step 1.
```
"""
function get_shared_dimensionality(subject::String, sessionidx::Int64,location::Int64; reg_window=(-400.0, -50.0), rtmin=120.0, rtmax=300.0, kvs...)
	h = CRC32c.crc32c(string(reg_window))
	h = CRC32c.crc32c(string(rtmin), h)
	h = CRC32c.crc32c(string(rtmax), h)
	for (k,v) in kvs
		h = CRC32c.crc32c(string(v), h)
	end
	qs = string(h, base=16)[1:8]
	fname = "$(subject)_session$(sessionidx)_location$(location)_shared_dimension_$(qs).jld2"
	if isfile(fname)
		ll, d_shared, ii = JLD2.load(fname, "loglikelihood","d_shared","max_dim")
	else
		ppstht, tlabelst, rtimest, trialidxt = JLD2.load("$(subject)_ppsth_cue_whole_trial.jld2", "ppsth","labels", "rtimes", "trialidx")
		X, bins, labels, rtimes = get_population_responses(ppstht, tlabelst, trialidxt, rtimest, sessionidx;rtmin=rtmin, rtmax=rtmax, mean_subtract=true, variance_stabilize=true)
		
		bidx = searchsortedfirst(bins, reg_window[1]):searchsortedlast(bins, reg_window[2])
		Xl = dropdims(mean(X[bidx, labels.==location,:],dims=1),dims=1)
		Xl = permutedims(Xl, [2,1])
		ll, d_shared, ii = get_shared_dimensionality(Xl;kvs...)
		JLD2.save(fname, Dict("loglikelihood" => ll,
							  "d_shared" => d_shared,
							  "max_dim" => ii))
	end
	ll, d_shared, ii	
end

"""
```julia
function get_shared_dimensionality(Xl::Matrix{Float64};method=:cm)
```

Get the dimensionality of the matrix `Xl` using factor analysis.
"""
function get_shared_dimensionality(Xl::Matrix{Float64};method=:cm)
	m = dropdims(mean(Xl,dims=2),dims=2)
	# remove cells with zero variance
	S = cov(Xl, dims=2)
	σ² = diag(S)
	pp = StatsBase.fit(Gamma, σ²)
	ppq = quantile(pp, 0.01)
	cidx = findall(σ² .> ppq)
	Xlc = Xl[cidx,:]
	n = size(Xl,2)
	m = m[cidx]
	ll = fill(0.0, size(Xlc,1))
	Threads.@threads for i in 1:length(ll)
		try
			for j in 1:n
				fa = StatsBase.fit(MultivariateStats.FactorAnalysis,Xlc[:,setdiff(1:n,j)];maxoutdim=i, method=method)
				dd = Xlc[:,j:j]-m
				Sd = dd*dd'
				_ll = loglikelihood(Sd, m, 1, fa)
				if isnan(_ll)
					@debug findall(isnan.(fa.Ψ)) findall(isnan.(fa.W))
					error("NaN value found for maxoutdim $i, j=$j")
				end
				ll[i] += _ll
			end
			ll[i] /= n
		catch ee
			if (ee.msg != "Eigenvalues less than 1") && !isa(ee, DomainError)
				rethrow(ee)
			end
		end
	end
	fidx = findall(isfinite, ll)
	if isempty(fidx)
		error("No finite values found")
	end
	ii = argmax(ll[fidx])
	fa = StatsBase.fit(MultivariateStats.FactorAnalysis,Xl;maxoutdim=fidx[ii], method=method)
	L = loadings(fa)
	#shared covariance
	
	Σs = L*L'
	d_shared = missing
	try
		u,s,v = svd(Σs)
		d_shared = findfirst(cumsum(s)/sum(s) .> 0.95)
	catch ee
		@debug Σs
		d_shared = missing
	end
	
	ll, d_shared, ii	
end

"""
```julia
function loglikelihood(S::AbstractMatrix{T}, mv::Vector{T}, n::Int64, fa::MultivariateStats.FactorAnalysis{T}) where T <: Real

Compute the log-likelihood of the data, represented by the sample covariance matrix `S` and the mean `mv`, over `n` observations given the factor model `fa`.
```
"""
function loglikelihood(S::AbstractMatrix{T}, mv::Vector{T}, n::Int64, fa::MultivariateStats.FactorAnalysis{T}) where T <: Real
	d = size(S,1)
	Ψ = fa.Ψ
	W = fa.W
	Ψ⁻¹= diagm(0 => 1 ./ Ψ)
	WᵀΨ⁻¹ = W'*Ψ⁻¹
	logdetΣ = sum(log, 	Ψ) + logabsdet(I + WᵀΨ⁻¹*W)[1]
	Σ⁻¹ = Ψ⁻¹ - Ψ⁻¹*W*inv(I + WᵀΨ⁻¹*W)*WᵀΨ⁻¹
	L = (-n/2)*(d*log(2π) + logdetΣ + tr(Σ⁻¹*S))
end
