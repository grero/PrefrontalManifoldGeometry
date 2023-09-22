module Figure2
using EventOnsetDecoding
using HypothesisTests
using MultivariateStats
using DataProcessingHierarchyTools
using JLD2
using HDF5
using CRC32c
using ColorSchemes
const DPHT = DataProcessingHierarchyTools

include("regression.jl")
include("plot_utils.jl")

using CairoMakie

sessions_j = ["J/20140807/session01", "J/20140828/session01", "J/20140904/session01", "J/20140905/session01"]
sessions_w = ["W/20200106/session02", "W/20200108/session03", "W/20200109/session04", "W/20200113/session01", "W/20200115/session03", "W/20200117/session03", "W/20200120/session01", "W/20200121/session01"]
sessions_p = ["P/20130923/session01", "P/20130927/session01", "P/20131014/session01", "P/20131021/session01"]
ramping_cells_j =  [22, 23, 24, 35, 52, 53, 58, 59, 60, 88, 89, 90, 91, 92, 103, 112, 113, 117]
ramping_cells_w =  [5, 24, 47, 54, 82]

function plot_fef_cell(cellidx::Int64,args...;kvs...)
    height = 5.0*72
    if get(Dict(kvs), :show_target, false)
        width = 1.6*height
    else
        width = 1.3*height
    end
    fig = Figure(resolution=(width,height))
    lg = GridLayout()
    fig[1,1] = lg
    plot_fef_cell!(lg, cellidx, args...;kvs...)
    fig
end

"""
    plot_fef_cell(cellidx::Int64, subject::String;kvs...)

Plot all locations for the specified cell
"""
function plot_fef_cell(cellidx::Int64, subject::String;kvs...)
    nlocations = length(locations[subject])
    height = 25.0*72
    if get(Dict(kvs), :show_target, false)
        width = 0.8*height
    else
        width = 0.5*height
    end
    fig = Figure(resolution=(width,height))
    axes = Any[]
    for l in 1:nlocations
        lg = GridLayout()
        fig[l,1] = lg
        if l < nlocations
            xticklabelsvisible = false
            xlabelvisible = false
        else
            xticklabelsvisible = true 
            xlabelvisible = true
        end
        ax = plot_fef_cell!(lg, cellidx, subject, collect(locations[subject][l:l]);xticklabelsvisible=xticklabelsvisible, xlabelvisible=xlabelvisible,kvs...)
        if ax === nothing
            return nothing
        end
        push!(axes, ax)
    end
    linkyaxes!(axes...)
    fig
end


"""
    get_cell_data(ppsth, cellidx::Int64, subject::String)

Get the data for cell `cellidx` coming from `subject`
"""
function get_cell_data(ppsth, trialidx::Vector{Vector{Int64}}, tlabel::Vector{Vector{Int64}}, rtimes::Dict{String, Vector{Float64}}, cellidx::Int64, subject::String;rtime_min=120.0, rtime_max=300.0)
    subject_idx = findall(c->DPHT.get_level_name("subject",c)==subject,ppsth.cellnames)
    if cellidx > length(subject_idx)
        return nothing
    end
    cellidx = subject_idx[cellidx]
    session = DPHT.get_level_path("session", ppsth.cellnames[cellidx])
    _rtimes = rtimes[session][trialidx[cellidx]]
    rtidx = findall(rtime_min .< _rtimes .< rtime_max)
    _rtimes = _rtimes[rtidx]
    _rtimes = rtimes[session][trialidx[cellidx]]
    rtidx = findall(rtime_min .< _rtimes .< rtime_max)
    _rtimes = _rtimes[rtidx]
    X = ppsth.counts[:,rtidx,cellidx]
    X, ppsth.bins, tlabel[cellidx][rtidx], _rtimes
end

function get_cell_data(alignment::String, args...;suffix="", kvs...)
    fnames = joinpath("data","ppsth_fef_$(alignment)_raw$(suffix).jld2")
    ppsth = JLD2.load(fnames, "ppsth")
    rtimes = JLD2.load(fnames, "rtimes")
    trialidx = JLD2.load(fnames, "trialidx")
    tlabel = JLD2.load(fnames, "labels")
    get_cell_data(ppsth, trialidx, tlabel, rtimes, args...;kvs...)
end

function plot_fef_cell!(fig, cellidx::Int64, subject::String, locations::Union{Vector{Int64}, Nothing}=nothing;rtime_min=120, rtime_max=300, windowsize=35.0, latency=0.0, latency_ref=:mov, 
                    tmin=(cue=-Inf, mov=-Inf,target=-Inf), tmax=(cue=Inf, mov=Inf, target=Inf), show_target=false, ylabelvisible=true, xlabelvisible=true, xticklabelsvisible=true,showmovspine=true,suffix="")
    #TODO: Plot all PSTH in one panel, with the raster per location stacked below
    #movement aligned
    fnames = joinpath("data","ppsth_fef_mov_raw$(suffix).jld2")
    ppsths = JLD2.load(fnames, "ppsth")
    rtimess = JLD2.load(fnames, "rtimes")
    trialidxs = JLD2.load(fnames, "trialidx")
    tlabels = JLD2.load(fnames, "labels")
    binss = ppsths.bins


    # cue aligned
    fnamec = joinpath("data","ppsth_fef_cue_raw$(suffix).jld2")
    ppsthc = JLD2.load(fnamec, "ppsth")
    rtimesc = JLD2.load(fnamec, "rtimes")
    trialidxc = JLD2.load(fnamec, "trialidx")
    tlabelc = JLD2.load(fnamec, "labels")
    binsc = ppsthc.bins

    # add target aligned here
    fnamet = joinpath("data","ppsth_fef_target_raw$(suffix).jld2")
    ppstht = JLD2.load(fnamet, "ppsth")
    rtimest = JLD2.load(fnamet, "rtimes")
    trialidxt = JLD2.load(fnamet, "trialidx")
    tlabelt = JLD2.load(fnamet, "labels")
    binst = ppstht.bins

    subject_idx = findall(c->DPHT.get_level_name("subject",c)==subject,ppsths.cellnames)
    if cellidx > length(subject_idx)
        return nothing
    end
    cellidx = subject_idx[cellidx]
    @assert ppsths.cellnames == ppsthc.cellnames
    # we have more cells for target aligned, so grab the subset that is also in ppsthc
    cellidxt = findfirst(ppstht.cellnames.==ppsthc.cellnames[cellidx])
    @assert trialidxs[cellidx] == trialidxc[cellidx]

    @assert length(trialidxt[cellidxt]) >= length(trialidxc[cellidx])
    ttidx = findall(in(trialidxc[cellidx]), trialidxt[cellidxt])
    @assert trialidxt[cellidxt][ttidx] == trialidxc[cellidx]
    session = DPHT.get_level_path("session", ppsths.cellnames[cellidx])

    _rtimes = rtimess[session][trialidxs[cellidx]]
    rtidx = findall(rtime_min .< _rtimes .< rtime_max)
    _rtimes = _rtimes[rtidx]
    if locations === nothing
        locations = unique(tlabels[cellidx][rtidx])
        sort!(locations)
    end
    tidx = findall(in(locations).(tlabels[cellidx][rtidx]))
    Xs = ppsths.counts[:,rtidx[tidx],cellidx]
    Xc = ppsthc.counts[:,rtidx[tidx],cellidx]
    Xt = ppstht.counts[:,ttidx[rtidx[tidx]],cellidxt]

    X2s,bins2s = rebin2(Xs, ppsths.bins, windowsize)
    X2c,bins2c = rebin2(Xc, ppsthc.bins, windowsize)
    X2t,bins2t = rebin2(Xt, ppstht.bins, windowsize)
    sidx = sortperm(-_rtimes[tidx])

    if latency_ref == :mov
        highlight_window = [_rtimes[tidx] .- latency .- windowsize, -latency - windowsize]
    else
        highlight_window = [latency, -_rtimes[tidx] .+ latency]
    end
        
    highlight_window
    if show_target
        ncols = 3
        align = [:target, :cue, :mov]
        _X = [Xt, Xc, Xs]
        _X2 = [X2t, X2c, X2s]
        bins = [binst, binsc, binss]
        bins2 = [bins2t, bins2c, bins2s]
        qq = [nothing, _rtimes[tidx], -_rtimes[tidx]]
        highlight_window = [nothing;highlight_window]
    else
        ncols = 2
        align = [:cue, :mov]
        _X = [Xc, Xs]
        _X2 = [X2c, X2s]
        bins = [binsc, binss]
        qq = [_rtimes[tidx], -_rtimes[tidx]]
        bins2 = [bins2c, bins2s]
    end

    with_theme(plot_theme) do
        axes = [Axis(fig[i,j], xticklabelsize=14, yticklabelsize=14) for i in 1:2, j in 1:ncols]
        linkyaxes!([axes[1,j] for j in 1:ncols]...)
        for (q, X, X2, bins, bins2, rtimesq, ww, ax1, ax2) in zip(align, _X, _X2, bins, bins2, qq, highlight_window, axes[1,:], axes[2,:])
            if ww !== nothing
                if length(ww) == 1
                    for ax in [ax1, ax2]
                        vspan!(ax, ww/1000.0, (ww .+ windowsize)./1000.0, color=RGB(0.8, 0.8, 1.0))
                    end
                else
                    xx = cat([[w,w+windowsize] for w in ww[sidx]]...,dims=1)
                    yy = cat([[i,i] for i in 1:length(ww[sidx])]...,dims=1)
                    linesegments!(ax2, xx./1000.0,yy, color=RGB(0.8, 0.8, 1.0), linewidth=3.0)
                end
            end
            vlines!(ax1, 0.0, color="black")
            bidx = searchsortedfirst(bins2, tmin[q]):searchsortedlast(bins2, tmax[q])
            
            lines!(ax1, bins2[bidx]/1000.0, 1000.0*dropdims(mean(X2[bidx,:], dims=2)/windowsize, dims=2))
            ax1.xticklabelsvisible = false
            ax1.xticksvisible = true
            ax1.yticksvisible = true
            
            linkxaxes!(ax1, ax2)
            bidx = searchsortedfirst(bins, tmin[q]):searchsortedlast(bins2, tmax[q])
            Xq =  X[bidx,sidx]
            qidx = findall(Xq .> minimum(Xq))
            #h = heatmap!(ax2, bins[bidx], 1:size(X,2), X[bidx,sidx], colormap=:Greys)
            scatter!(ax2, bins[bidx[[I.I[1] for I in qidx]]]./1000.0, [I.I[2] for I in qidx], markersize=10px, color="black", marker='|')
            vlines!(ax2, 0.0, color="black")
            if rtimesq !== nothing
                scatter!(ax2, rtimesq[sidx]./1000.0, 1:length(sidx), color="red", markersize=5px)
            end
            ax2.xticksvisible = true
            ax2.yticksvisible = true
            
        end
        for ax in axes[:,2:end]
            ax.yticklabelsvisible = false
        end
        axes[2,1].yticklabelsvisible = false
        if ylabelvisible
            axes[2,1].ylabel = "Trial ID"
            axes[1,1].ylabel = "Activity [Hz]"
        end
        if xlabelvisible
            if show_target
                axes[2,1].xlabel = "Target [ms]"
                axes[2,2].xlabel = "Go-cue [ms]"
                axes[2,3].xlabel = "Movement [ms]"
            else
                axes[2,1].xlabel = "Go-cue"
                axes[2,2].xlabel = "Movement"
            end
        end
        axes[2,1].xticklabelsvisible = xticklabelsvisible
        axes[2,2].xticklabelsvisible = xticklabelsvisible
        if !showmovspine
            for i in 1:2
                ax = axes[i,2]
                ax.yticksvisible=false
                ax.leftspinevisible=false
            end
        end
        rowgap!(fig,1,5)
        axes[1,1] 
    end
end

function plot_psth_and_raster(subject::String, cellidx::Int64, windowsize::Float64;suffix="",kvs...)
    # get the number of locations to determine the height
    fnames = joinpath("data","ppsth_fef_mov_raw$(suffix).jld2")
    tlabels = JLD2.load(fnames, "labels")
    ulabel = unique(tlabels)
    nl = length(ulabel)
    height = nl*100 
    fig = Figure(resolution=(1500,height))
    lg = GridLayout()
    fig[1,1] = lg
    plot_psth_and_raster!(lg, subject, cellidx, windowsize;suffix=suffix, kvs...)
    fig
end

function plot_psth_and_raster!(lg, subject::String, cellidx::Int64, windowsize::Float64;rtime_min=120.0, rtime_max=300.0, suffix="", locations::Union{Nothing, Vector{Int64}}=locations[subject],
                                                                                    tmin=(cue=-Inf, mov=-Inf,target=-Inf), tmax=(cue=Inf, mov=Inf, target=Inf),kvs...)
    #movement aligned
    fnames = joinpath("data","ppsth_fef_mov_raw$(suffix).jld2")
    ppsths = JLD2.load(fnames, "ppsth")
    rtimess = JLD2.load(fnames, "rtimes")
    trialidxs = JLD2.load(fnames, "trialidx")
    tlabels = JLD2.load(fnames, "labels")
    binss = ppsths.bins


    # cue aligned
    fnamec = joinpath("data","ppsth_fef_cue_raw$(suffix).jld2")
    ppsthc = JLD2.load(fnamec, "ppsth")
    rtimesc = JLD2.load(fnamec, "rtimes")
    trialidxc = JLD2.load(fnamec, "trialidx")
    tlabelc = JLD2.load(fnamec, "labels")
    binsc = ppsthc.bins

    # add target aligned here
    fnamet = joinpath("data","ppsth_fef_target_raw$(suffix).jld2")
    ppstht = JLD2.load(fnamet, "ppsth")
    rtimest = JLD2.load(fnamet, "rtimes")
    trialidxt = JLD2.load(fnamet, "trialidx")
    tlabelt = JLD2.load(fnamet, "labels")
    binst = ppstht.bins

    subject_idx = findall(c->DPHT.get_level_name("subject",c)==subject,ppsths.cellnames)
    if cellidx > length(subject_idx)
        return nothing
    end
    cellidx = subject_idx[cellidx]
    @assert ppsths.cellnames == ppsthc.cellnames
    # we have more cells for target aligned, so grab the subset that is also in ppsthc
    cellidxt = findfirst(ppstht.cellnames.==ppsthc.cellnames[cellidx])
    @assert trialidxs[cellidx] == trialidxc[cellidx]

    @assert length(trialidxt[cellidxt]) >= length(trialidxc[cellidx])
    ttidx = findall(in(trialidxc[cellidx]), trialidxt[cellidxt])
    @assert trialidxt[cellidxt][ttidx] == trialidxc[cellidx]
    session = DPHT.get_level_path("session", ppsths.cellnames[cellidx])

    _rtimes = rtimess[session][trialidxs[cellidx]]
    rtidx = findall(rtime_min .< _rtimes .< rtime_max)
    _rtimes = _rtimes[rtidx]
    if locations === nothing
        locations = unique(tlabels[cellidx][rtidx])
        sort!(locations)
    end
    tidx = findall(in(locations).(tlabels[cellidx][rtidx]))
    Xs = ppsths.counts[:,rtidx[tidx],cellidx]
    Xc = ppsthc.counts[:,rtidx[tidx],cellidx]
    Xt = ppstht.counts[:,ttidx[rtidx[tidx]],cellidxt]

    # one column for each alignment
    lgt = GridLayout()
    #fig[1,1] = lgt
    lg[1,1] = lgt
    lgc = GridLayout()
    #fig[1,2] = lgc
    lg[1,2] = lgc
    lgs = GridLayout()
    #fig[1,3] = lgs
    lg[1,3] = lgs
    tridx = rtidx[tidx]
    bidxt = searchsortedfirst(binst, tmin.target):searchsortedlast(binst, tmax.target)
    bidxc = searchsortedfirst(binsc, tmin.cue):searchsortedlast(binsc, tmax.cue)
    bidxs = searchsortedfirst(binss, tmin.mov):searchsortedlast(binss, tmax.mov)
    axt = plot_psth_and_raster!(lgt, Xt[bidxt, :], binst[bidxt], location_idx[subject][tlabelt[cellidxt][ttidx[tridx]]],_rtimes, windowsize;multiplier=nothing, xlabel="Target onset[ms]", kvs...)
    axc = plot_psth_and_raster!(lgc, Xc[bidxc, :], binsc[bidxc], location_idx[subject][tlabelc[cellidx][tridx]],_rtimes, windowsize;multiplier=1.0, ylabelvisible=false, yticklabelsvisible=false, xlabel="Go-cue onset[ms]", kvs...)
    axs = plot_psth_and_raster!(lgs, Xs[bidxs,:], binss[bidxs], location_idx[subject][tlabels[cellidx][tridx]],_rtimes, windowsize;multiplier=-1.0, ylabelvisible=false, yticklabelsvisible=false, xlabel="Movement onset [ms]", kvs...)
    linkyaxes!(axt, axc, axs)
    lg
end

"""
    plot_psth_and_raster(X::Matrix{T}, bins::AbstractVector{T},tlabel::Vector{Int64}, windowsize=1.0;xlabel="") where T <: Real

Plot the PSTH for each location in a single panel, followed by a stacking of the rasters for each location, color-coded
    similarly to the PSTH
"""
function plot_psth_and_raster(X::Matrix{T}, bins::AbstractVector{T},tlabel::Vector{Int64}, rtime::Vector{Float64}, windowsize=1.0;kvs...) where T <: Real
    ulabel = unique(tlabel)
    nl = length(ulabel)
    height = nl*500/4 
    fig = Figure(resolution=(500,height))
    lg = GridLayout()
    fig[1,1] = lg
    plot_psth_and_raster!(lg, X, bins, tlabel, rtime, windowsize;kvs...)
    fig
end

function plot_psth_and_raster!(lg, X::Matrix{T}, bins::AbstractVector{T},tlabel::Vector{Int64}, rtime::Vector{Float64}, windowsize=1.0;xlabel="", multiplier=1.0, ylabelvisible=true, yticklabelsvisible=true, xlabelvisible=true, xticklabelsvisible=true, toprowsize=250) where T <: Real
    X2,bins2 = rebin2(X,bins,windowsize)
    binsize = bins[2] - bins[1]
    nb,nt = size(X)
    ulabel = unique(tlabel)
    sort!(ulabel)
    nl = length(ulabel)
    μ = fill(0.0, length(bins2),nl)
    nq = fill(0,1, nl)
    for j in axes(X,2)
        lidx = findfirst(ulabel.==tlabel[j])
        μ[:,lidx] += X2[1:length(bins2),j]
        nq[lidx] += 1
    end
    μ ./= nq
    μ ./= (windowsize/1000.0)

    # raster
    Xmin = minimum(X)
    _colors = ColorSchemes.Paired_10.colors
    # since this scheme has 10 colors, index using number from 1 to 8
    # for W, the locations are the 4 corners
    # for J, the locations are the 4 corners plus the cardinals
    with_theme(plot_theme) do
        axp = Axis(lg[1,1])
        vlines!(axp, 0.0, color="black",linestyle=:dot)
        for i in axes(μ,2)
            lines!(axp, bins2, μ[:,i], color=_colors[ulabel[i]], linewidth=2.0)
        end
        axp.xticklabelsvisible = false
        axp.xticksvisible = false
        axp.bottomspinevisible = false
        axp.yticklabelsvisible = yticklabelsvisible
        if ylabelvisible
            axp.ylabel = "Firing rate [Hz]"
        end
        axesr = [Axis(lg[1+i,1]) for i in 1:nl]
        linkxaxes!(axesr..., axp)
        # raster
        for (ll, ax) in enumerate(axesr)
            # grab all trials with this label
            tidx = findall(tlabel.==ulabel[ll])
            sidx = sortperm(rtime[tidx])
            qidx = findall(X[:,tidx[sidx]] .> Xmin)
            # grap the spikes for this trial
            vlines!(ax, 0.0, color="black",linestyle=:dot)
            scatter!(ax, bins[[I.I[1] for I in qidx]], [I.I[2] for I in qidx], markersize=15px, color=_colors[ulabel[ll]], marker='|')
            # show rtime
            if multiplier !== nothing
                scatter!(ax, multiplier*rtime[tidx[sidx]], 1:length(sidx), color="black", marker='|')
            end
            ax.xticklabelsvisible = false 
            ax.bottomspinevisible = false
            ax.xticksvisible = false
            ax.yticksvisible = false
            ax.yticklabelsvisible = false
            ax.leftspinevisible = false
            rowgap!(lg, ll, 5.0)
        end

        ax = axesr[end]
        ax.xticksvisible = true
        ax.xticklabelsvisible = true
        ax.bottomspinevisible = true
        ax.xlabel = xlabel
        ax.xlabelvisible=xlabelvisible
        ax.xticklabelsvisible = xticklabelsvisible
        if ylabelvisible
            ax.ylabel = "TrialID"
        end
        #TODO: Make this a function argument.
        rowsize!(lg, 1, toprowsize)
        axp 
    end
end

"""
Get the explained reaction time variance as a function of time
"""
function get_rtime_var_per_time(;redo=false, do_save=true,kvs...)
    fname = "rtime_var_in_time.jld2"
    if isfile(fname) && !redo
        plot_data = JLD2.load(fname, "plot_data")
    else
        plot_data = Dict{Symbol,Any}()
        r2_rtime_mov, bins, rtlims, r²j = regress_reaction_time(;align=:mov,nruns=1000,kvs...);
        r2_rtime_mov_shuffled, _, _, r²j_shuffled = regress_reaction_time(;align=:mov,do_shuffle=true, nruns=1000,kvs...);
        y = r2_rtime_mov
        ys = r2_rtime_mov_shuffled
        l,m,u = [fill(0.0, size(y,1)) for i in 1:3]
        ls,ms,us = [fill(0.0, size(y,1)) for i in 1:3]
        for i in 1:size(y,1)
            l[i],m[i], u[i] = percentile(y[i,:], [5,50,95])
        end
        for i in 1:size(y,1)
            ls[i],ms[i], us[i] = percentile(ys[i,:], [5,50,95])
        end
        r2j = r²j 
        r2js = r²j_shuffled 
        pv = fill(0.0, size(r2j,1))
        for i in 1:length(pv)
            pv[i] = pvalue(HypothesisTests.MannWhitneyUTest(r2j[i,:], r2js[i,:]);tail=:right)
        end
        # create indices for highlight
        xh = Tuple{Float64, Float64}[]
        x0 = -Inf 
        for i in 1:length(pv)
            if pv[i] < 0.01
                if x0 == -Inf
                    x0 = bins[i]
                end
            else
                if x0 > -Inf
                    push!(xh, (x0, bins[i]))
                    x0 = -Inf
                end
            end
        end
        plot_data[:pv] = pv
        plot_data[:xh] = xh
        plot_data[:rtlims] = rtlims
        plot_data[:bins] = bins
        plot_data[:original] = Dict(:m=>m, :l=>l, :u=>u, :r2=>r2_rtime_mov, :r2j=>r²j)
        plot_data[:shuffled] = Dict(:m=>ms, :l=>ls, :u=>us, :r2=>r2_rtime_mov_shuffled, :r2j=>r²j_shuffled)
        if do_save
            JLD2.save(fname, Dict("plot_data" => plot_data))
        end
    end
    plot_data
end

"""
    remove_window!(ppsth, window::Tuple{Float64, Float64}, t0::Float64=0.0)

Remov the responses in the specified window by replacing them with responses from the
previous window.

#Examples
```jldoctest
ppsth = (counts=repeat([0 1;1 1;1 0;0 0],1,1,1), bins=[1.0,2.0,3.0,4.0],
         cellname=["A/2014094/session01/array01/channel001/cell01"])
remove_window!(ppsth, (3.0, 4.0))
ppsth.counts

# output

4×2×1 Array{Int64, 3}:
[:, :, 1] =
 0  1
 1  1
 0  1
 1  1
```
"""
function remove_window!(ppsth, window::Tuple{Float64, Float64}, t0::Float64=0.0)
    bins = ppsth.bins
    idx1 = searchsortedfirst(bins, t0+window[1])
    idx2 = searchsortedlast(bins, t0+window[2])
    w = idx2-idx1+1
    ppsth.counts[idx1:idx2,:,:] .= ppsth.counts[idx1-w:idx1-1,:,:]
    ppsth
end

"""
    remove_window!(ppsth,trialidx::Vector{Vector{Int64}}, window::Tuple{Float64, Float64}, t0::Dict{String, Vector{Float64}},ss::Float64=1.0)

Remove responses in the specified window using the reference `t0`.

# Examples
```jldoctest
ppsth = (counts=repeat([0 0;1 1;0 0;1 1;0 0],1,1,1), bins=[1.0,2.0,3.0,4.0,5.0],
         cellname=["A/20140904/session01/array01/channel001/cell01"])
rtime = Dict("A/20140904/session01"=>[1.0, 2.0])

remove_window!((ppsth, [[1,2]], (5.0, 5.0), rtime, -1.0)
ppsth.counts

# output

5×2×1 Array{Int64, 3}:
[:, :, 1] =
 0  0
 1  1
 0  1
 0  1
 0  0
 ```
"""
function remove_window!(ppsth,trialidx::Vector{Vector{Int64}}, window::Tuple{Float64, Float64}, t0::Dict{String, Vector{Float64}},ss::Float64=1.0)
    bins = ppsth.bins
    cells = EventOnsetDecoding.get_cellnames(ppsth)
    for (ic,cell) in enumerate(cells)
        session = DPHT.get_level_path("session", cell)
        for (i,_t0) in enumerate(t0[session][trialidx[ic]])
            idx1 = searchsortedfirst(bins, ss*_t0+window[1])
            idx2 = searchsortedlast(bins, ss*_t0+window[2])
            w = idx2-idx1+1
            if idx1-w > 0
                ppsth.counts[idx1:idx2,i,ic] .= ppsth.counts[idx1-w:idx1-1,i,ic]
            end
        end
    end
    ppsth
end

function get_event_subspaces(;nruns=100,area="FEF",redo=false,combine_locations=true,subject="ALL",
                              rtime_min=120.0, remove_window::Union{Nothing, Dict{Symbol, Tuple{Float64, Float64}}}=nothing, save_sample_indices::Bool=false,suffix="")
    sarea = lowercase(area)
    ppsth_mov,labels_mov, trialidx_mov, rtimes_mov = JLD2.load("data/ppsth_$(sarea)_mov$(suffix).jld2","ppsth", "labels","trialidx","rtimes")
    ppsth_cue,labels_cue, trialidx_cue, rtimes_cue = JLD2.load("data/ppsth_$(sarea)_cue$(suffix).jld2","ppsth", "labels","trialidx","rtimes")
    cellidx = get_area_index(ppsth_mov.cellnames, area)
    @assert get_area_index(ppsth_cue.cellnames, area) == cellidx

    h = zero(UInt32)
    if remove_window !== nothing
        for (window_ref,v) in remove_window
            h = crc32c("remove_window=$(v)",h)
            h = crc32c("window_ref=$(window_ref)",h)
            if window_ref == :cue
                remove_window!(ppsth_cue, v)
                remove_window!(ppsth_mov, trialidx_mov, v, rtimes_mov, -1.0)
            else
                remove_window!(ppsth_mov, v)
                remove_window!(ppsth_cue, trialidx_cue, v, rtimes_cue, 1.0)
            end
        end
    end
    cellidx = findall(cellidx)
    if subject == "ALL"
        sessions = [sessions_j;sessions_w]
        locations = [1:8;]
    elseif subject == "W"
        sessions = sessions_w
        locations = [1:4;]
    elseif subject == "J"
        sessions = sessions_j
        locations = [1:8;]
    elseif subject == "P"
        sessions = sessions_p
        locations = [1:8;]
    else
        error("Unknown subject $(subject). Should be one of ALL, W, or J")
    end
	args = [sessions, locations, cellidx]
	latencies = range(100.0, step=-10.0, stop=0.0)
	windows = [range(5.0, step=10.0, stop=50.0);]
	kvs = [:nruns=>nruns, :difference_decoder=>true, :windows=>windows,
		:latencies=>latencies, :combine_locations=>combine_locations, :use_area=>"ALL",
		:rtime_min=>rtime_min, :mixin_postcue=>true,
		:shuffle_bins=>false, :shuffle_latency=>false, :simple_shuffle=>false,
		:restricted_shuffle=>false, :shuffle_each_trial=>false, :fix_resolution=>false,
		:at_source=>false, :shuffle_training=>false, :use_new_decoder=>false,
		:combine_training_only=>false, :max_shuffle_latency=>50.0, :save_sample_indices=>save_sample_indices,
		]
	dargs_cue = EventOnsetDecoding.DecoderArgs(args...;kvs..., reverse_bins=true, baseline_end=-250.0)
    fname_cue = EventOnsetDecoding.get_filename(dargs_cue)
	dargs_mov = EventOnsetDecoding.DecoderArgs(args...;kvs..., reverse_bins=false, baseline_end=-300.0)
    fname_mov = EventOnsetDecoding.get_filename(dargs_mov)

    @show fname_cue fname_mov
    rseeds = rand(UInt32, dargs_mov.nruns)

    @show size(ppsth_cue.counts) size(trialidx_cue) size(labels_cue) collect(keys(rtimes_cue))
    perf_cue,rr_cue,f1score_cue,fname_cue = EventOnsetDecoding.run_rtime_decoder(ppsth_cue,trialidx_cue,labels_cue,rtimes_cue,dargs_cue,
                                                             ;decoder=MultivariateStats.MulticlassLDA, rseeds=rseeds, redo=redo, h_init=h)
    perf_mov,rr_mov,f1score_mov,fname_mov = EventOnsetDecoding.run_rtime_decoder(ppsth_mov,trialidx_mov,labels_mov,rtimes_mov,dargs_mov,
                                                             ;decoder=MultivariateStats.MulticlassLDA, rseeds=rseeds, redo=redo, h_init=h)

    fname_cue, fname_mov
end
 

function plot_event_onset_subspaces(fname_cue, fname_mov;width=700, height=400, kvs...)
    with_theme(plot_theme) do
        fig = Figure(resolution=(width,height))
        lg = GridLayout()
        fig[1,1] = lg
        plot_data = plot_event_onset_subspaces!(lg, fname_cue, fname_mov;kvs...)
        fig
    end
end

function plot_performance(f1score::Array{Float64,3},args...;kvs...)
    fig = Figure(resolution=(500,300))
    plot_performance!(fig, f1score, args...;kvs...)
    fig
end

function plot_performance!(lg, f1score::Array{Float64,3}, windows::AbstractVector{Float64}, latencies::AbstractVector{Float64};α::StatsBase.PValue=StatsBase.PValue(0.001), threshold=0.5, show_colorbar=true, kvs...)
    ax = Axis(lg[1,1])
    cidx = findall(dropdims(maximum(x->isnan(x) ? -Inf : x, f1score, dims=(1,2)),dims=(1,2)).>0.0)
    lower_limit = fill(0.0, length(windows), length(latencies))
    μ = fill(0.0, length(windows), length(latencies))
    for iw in 1:length(windows)
        for il in 1:length(latencies)
            dd = fit(Beta, filter(isfinite, f1score[iw,il,cidx]))
            lower_limit[iw,il] = quantile(dd, α.v)
            μ[iw,il] = mean(dd)
        end
    end
    pidx = findall(lower_limit .> threshold) 
    h = heatmap!(ax, windows, latencies, μ;kvs...)
    scatter!(ax, windows[[p.I[1] for p in pidx]], latencies[[p.I[2] for p in pidx]], marker='*', color="red", markersize=20px)
    if show_colorbar
        l = Colorbar(lg[1,2], h)
    end
    ax.xticks = windows
    h,ax
end

function plot_event_onset_subspaces!(lg0, fname_cue, fname_mov;max_latency=Inf, α=0.001, threshold=0.5, ymin=Inf,
                                                               ymax=-Inf, show_colorbar=true,kvs...)
    plot_data = Dict{Symbol, Any}(:cue => Dict{Symbol, Any}(), :mov => Dict{Symbol, Any}())
    for (k,fname) in zip([:cue, :mov], [fname_cue, fname_mov])
        bins, f1score,windows,latencies = h5open(fname) do fid
            read(fid, "bins"), read(fid, "f1score"), read(fid,"window"), read(fid, "latency")
        end
        cidx = findall(dropdims(maximum(x->isnan(x) ? -Inf : x, f1score, dims=(1,2,3)),dims=(1,2,3)).>0.0)
        lower_limit = fill(0.0, length(windows), length(latencies))
        μ = fill(0.0, length(windows), length(latencies))
        for iw in 1:length(windows)
            for il in 1:length(latencies)
                dd = fit(Beta, filter(isfinite, f1score[iw,il,1, cidx]))
                lower_limit[iw,il] = quantile(dd, α)
                μ[iw,il] = mean(dd)
            end
        end
        ymin = min(minimum(μ), ymin)
        ymax = max(maximum(μ), ymax)
        pidx = findall(lower_limit .> threshold)
        plot_data[k][:μ] = μ
        plot_data[k][:lower_limit] = lower_limit
        plot_data[k][:pidx] = pidx
        plot_data[k][:windows] = windows
        plot_data[k][:latencies] = latencies
    end
    plot_data[:ymin] = ymin
    plot_data[:ymax] = ymax
    with_theme(plot_theme) do
        axes = [Axis(lg0[1,i]) for i in 1:2]
        ymin = plot_data[:ymin]
        ymax = plot_data[:ymax]
        for (k,ax) in zip([:cue, :mov],axes)
            windows = plot_data[k][:windows]
            latencies = plot_data[k][:latencies]
            _max_latency = min(max_latency, maximum(latencies))
            lidx = searchsortedlast(latencies, _max_latency, rev=true):length(latencies)
            μ = plot_data[k][:μ]
            lower_limit = plot_data[k][:lower_limit]
            pidx = plot_data[k][:pidx]
            qidx = pidx[(in(lidx)).([p.I[2] for p in pidx])]
            @show lidx latencies max_latency
            h = heatmap!(ax, windows, latencies[lidx], μ[:,lidx],  colormap=:Blues,colorrange=(ymin, ymax))
            if k == :mov && show_colorbar
                # TODO: Maybe put this below instead of at the side?
                cb = Colorbar(lg0[1,3], h, label="F1-score",ticklabelsize=12)
            end
            # highlight some bins
            if k == :cue
                pp = get_rectangular_border(10.0, 35.0, 20.0, 45.0)
                lines!(ax, pp, color="red")
            else
                pp = get_rectangular_border(30.0, -5.0, 40.0, 5.0)
                lines!(ax, pp, color="red")
            end
            scatter!(ax, windows[[p.I[1] for p in qidx]], latencies[[p.I[2] for p in qidx]], marker='*', color="red", markersize=20px)
            ax.xticks = windows
        end
        ax = axes[1]
        ax.xlabel = "Window size [ms]"
        if get(Dict(kvs), :ylabelvisible, true)
            ax.ylabel = "Latency [ms]"
        end
        ax.yticklabelsvisible = get(Dict(kvs), :yticklabelsvisible, true)
        ax.title = "Go-cue onset"
        ax.titlefont = :regular

        ax = axes[2]
        ax.yticklabelsvisible = false
        ax.title = "Movement onset"
        ax.titlefont = :regular
        lg0
    end
    plot_data
end


function plot(;do_save=false,max_latency=Inf, width=900, height=500, kvs...)
    fname = "fig2_data.jld2"
    α = 0.001
    threshold = 0.5
    if isfile(fname)
        plot_data, plot_data_reg = JLD2.load(fname, "plot_data", "plot_data_reg")
    else
        plot_data = Dict{Symbol, Any}(:cue => Dict{Symbol, Any}(), :mov => Dict{Symbol, Any}())
        ymin,ymax = (Inf, -Inf)
        fname_cue, fname_mov = get_event_subspaces(;nruns=100)
        for (k,fname) in zip([:cue, :mov], [fname_cue, fname_mov])
            bins, f1score,windows,latencies = h5open(fname) do fid
                read(fid, "bins"), read(fid, "f1score"), read(fid,"window"), read(fid, "latency")
            end
            cidx = findall(dropdims(maximum(x->isnan(x) ? -Inf : x, f1score, dims=(1,2,3)),dims=(1,2,3)).>0.0)
            lower_limit = fill(0.0, length(windows), length(latencies))
            μ = fill(0.0, length(windows), length(latencies))
            for iw in 1:length(windows)
                for il in 1:length(latencies)
                    dd = fit(Beta, filter(isfinite, f1score[iw,il,1, cidx]))
                    lower_limit[iw,il] = quantile(dd, α)
                    μ[iw,il] = mean(dd)
                end
            end
            ymin = min(minimum(μ), ymin)
            ymax = max(maximum(μ), ymax)
            pidx = findall(lower_limit .> threshold)
            plot_data[k][:μ] = μ
            plot_data[k][:lower_limit] = lower_limit
            plot_data[k][:pidx] = pidx
            plot_data[k][:windows] = windows
            plot_data[k][:latencies] = latencies
        end
        plot_data[:ymin] = ymin
        plot_data[:ymax] = ymax

        plot_data_reg = get_rtime_var_per_time()
        JLD2.save(fname, Dict("plot_data"=>plot_data, "plot_data_reg"=>plot_data_reg))
    end
    with_theme(plot_theme) do
        fig = Figure(resolution=(width,height))
        lg0 = GridLayout()
        fig[1,1] = lg0
        axes = [Axis(lg0[1,i]) for i in 1:2]
        ymin = plot_data[:ymin]
        ymax = plot_data[:ymax]
        for (k,ax) in zip([:cue, :mov],axes)
            windows = plot_data[k][:windows]
            latencies = plot_data[k][:latencies]
            lidx = searchsortedfirst(latencies, max_latency, rev=true):length(latencies)
            μ = plot_data[k][:μ]
            lower_limit = plot_data[k][:lower_limit]
            pidx = plot_data[k][:pidx]
            h = heatmap!(ax, windows, latencies[lidx], μ[:,lidx],  colormap=:Blues,colorrange=(ymin, ymax))
            if k == :mov
                # TODO: Maybe put this below instead of at the side?
                cb = Colorbar(lg0[1,3], h, label="F1-score",ticklabelsize=12)
            end
            # highlight some bins
            if k == :cue
                pp = get_rectangular_border(10.0, 35.0, 20.0, 45.0)
                lines!(ax, pp, color="red")
            else
                pp = get_rectangular_border(30.0, -5.0, 40.0, 5.0)
                lines!(ax, pp, color="red")
            end
            qidx = pidx[(in(lidx)).([p.I[2] for p in pidx])]
            scatter!(ax, windows[[p.I[1] for p in qidx]], latencies[[p.I[2] for p in qidx]], marker='*', color="red", markersize=20px)
            ax.xticks = windows
        end
        ax = axes[1]
        ax.xlabel = "Window size [ms]"
        ax.ylabel = "Latency [ms]"
        ax.title = "Go-cue onset"
        ax.titlefont = :regular
        ax = axes[2]
        ax.yticklabelsvisible = false
        ax.title = "Movement onset"
        ax.titlefont = :regular
        # plot cells
        lg = GridLayout()
        fig[1,2] = lg
        Label(lg[1, 1, Top()], "Go-cue decoder", tellwidth=false, tellheight=false)
        Label(lg[1, 2, Top()], "Movement decoder", tellwidth=false, tellheight=false)
        showmovspine = false
        lg1 = GridLayout()
        lg[1,2] = lg1
        plot_fef_cell!(lg1, 22,"J", [6];windowsize=35.0, latency=0.0, latency_ref=:mov, tmin=(cue=-100, mov=-250), tmax=(cue=250, mov=50),xlabelvisible=false, xticklabelsvisible=false, ylabelvisible=false, showmovspine=showmovspine)
        rowgap!(lg1, 1, 4.0)
        colgap!(lg1, 1, 4.0)
        lg2 = GridLayout()
        lg[2,2] = lg2
        plot_fef_cell!(lg2, 3,"J", [6];windowsize=35.0, latency=0.0, latency_ref=:mov, tmin=(cue=-100, mov=-250), tmax=(cue=250, mov=50),xlabelvisible=false, xticklabelsvisible=false, ylabelvisible=false, showmovspine=showmovspine)
        rowgap!(lg2, 1, 4.0)
        colgap!(lg2, 1, 4.0)
        lg3 = GridLayout()
        lg[1,1] = lg3
        # cue aligned
        plot_fef_cell!(lg3, 28,"W", [2];windowsize=15.0, latency=50.0, latency_ref=:cue, tmin=(cue=-100, mov=-250), tmax=(cue=250, mov=50),xlabelvisible=false, xticklabelsvisible=false, ylabelvisible=false, showmovspine=showmovspine)
        rowgap!(lg3, 1, 4.0)
        colgap!(lg3, 1, 4.0)
        lg4 = GridLayout()
        lg[2,1] = lg4
        plot_fef_cell!(lg4, 59,"J", [6];windowsize=15.0, latency=50.0, latency_ref=:cue, tmin=(cue=-100, mov=-250), tmax=(cue=250, mov=50), ylabelvisible=true, showmovspine=showmovspine, xlabelvisible=true, xticklabelsvisible=true)
        rowgap!(lg4, 1, 4.0)
        colgap!(lg4, 1, 4.0)
        colgap!(lg, 1, 5.0)
    
    
        # regression panel
        ax5 = Axis(lg0[2,1:3])
    
        bins = plot_data_reg[:bins]
        bidx = searchsortedfirst(bins, -200.0):searchsortedlast(bins, 50.0)
        # transition period in blue?
        vspan!(ax5, -203.0 + 65.0, -35.0,color=RGB(0.9, 0.9, 1.0))
        for (x0,x1) in plot_data_reg[:xh]
            vspan!(ax5, x0, x1,color=RGB(0.8, 0.8, 0.8))
        end
        lines!(ax5, bins[bidx], dropdims(mean(plot_data_reg[:original][:r2j],dims=2),dims=2)[bidx], label="Data")
        lines!(ax5, bins[bidx], dropdims(mean(plot_data_reg[:shuffled][:r2j],dims=2),dims=2)[bidx], label="Surrogate")
        axislegend(ax5, halign=:right, valign=:top, margin=(10.0, 0.0, 40.0, -50.0))
        ax5.xlabel = "Time from movemenet [ms]"
        ax5.ylabel = L"adjusted $r^2$"
    
        label_padding = (0.0, 0.0, 5.0, 0.0)
        labels = [Label(lg0[1,1,TopLeft()],"A", padding=label_padding),
                  Label(lg[1,1,TopLeft()], "B", padding=label_padding),
                  Label(lg[1,2,TopLeft()], "C", padding=label_padding),
                  Label(lg0[2,1,TopLeft()],"D", padding=label_padding)]
        colgap!(fig.layout,1,1.0)
        colsize!(fig.layout, 1, Relative(0.4))
        fname = joinpath("figures","manuscript", "subspaces_and_regression.pdf")
        if do_save
            save(fname, fig;pt_per_unit=1)
        end
        fig
    end
end

visual_cells = [("W", 45),("W", 40)] 
movement_cells = [("W", 54),("J", 22)]
visuo_movement_cells = [("W", 8),("J", 53)]
post_saccadic_cells = [("W", 93), ("J", 5)]
movement_decreasing_cells = [("W", 51),("J", 39)]
go_cue_cells = [("J", 92), ("J", 59)]

plot_single_cell_examples(windowsize::Real;kvs...) = plot_single_cell_examples([("W", 45),
                                                                                ("W", 40),
                                                                                ("W", 54),
                                                                                ("J", 22),
                                                                                ("W", 8),
                                                                                ("J", 53)],
                                                                                windowsize;kvs...)

function plot_single_cell_examples(cells::Vector{Tuple{String, Int64}}, windowsize;labels::Vector{String}=["A","B","C"], kvs...)
    width = 15*2.5*72
    height = width
    with_theme(plot_theme) do
        fig = Figure(resolution=(width,height))

        # visual cells
        lg11 = GridLayout()
        fig[1,1] = lg11
        plot_psth_and_raster!(lg11, cells[1][1], cells[1][2],windowsize;xlabelvisible=false, xticklabelsvisible=false, kvs...)

        lg12 = GridLayout()
        fig[1,2] = lg12
        plot_psth_and_raster!(lg12, cells[2][1], cells[2][2],windowsize;ylabelvisible=false, xlabelvisible=false, xticklabelsvisible=false, kvs...)

        # movement cells
        lg21 = GridLayout()
        fig[2,1] = lg21
        plot_psth_and_raster!(lg21, cells[3][1], cells[3][2],windowsize;ylabelvisible=true, xlabelvisible=false, xticklabelsvisible=false, kvs...)

        lg22 = GridLayout()
        fig[2,2] = lg22
        plot_psth_and_raster!(lg22, cells[4][1], cells[4][2],windowsize;ylabelvisible=false, xlabelvisible=false, xticklabelsvisible=false, kvs...)

        # visuo-movement cells
        lg31 = GridLayout()
        fig[3,1] = lg31
        plot_psth_and_raster!(lg31, cells[5][1], cells[5][2],windowsize;kvs...)

        lg32 = GridLayout()
        fig[3,2] = lg32
        plot_psth_and_raster!(lg32, cells[6][1], cells[6][2],windowsize;ylabelvisible=false, kvs...)
        lmargin = (20,20,20,20)
        labels = [Label(fig[i,1, TopLeft()], label, padding=lmargin) for (i,label) in enumerate(labels)]
        fig
    end
end
end # module