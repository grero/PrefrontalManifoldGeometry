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
using OrthogonalLDA
using EventOnsetDecoding
using DataProcessingHierarchyTools

const DPHT = DataProcessingHierarchyTools

include("utils.jl")
include("plot_utils.jl")
"""
````
function find_orthogonal_subpspaces(subject;redo=false, do_pca=false,
````
The a orthogonal sub-spaces for for activity in the specified windows
"""
function find_orthogonal_subpspaces(;redo=false, do_pca=false,
    nruns=20,
    windowsize=40.0,
    window1=(-600.0, 0.0), 
    window2=(0.0, 400.0),
    window3=(-100.0, 0.0),
    window4=(0.0, 100.0),
    use_locations::Union{Vector{Int64},Symbol}=:all, use_corners=false)

    h = CRC32c.crc32c(string(window1))
    h = CRC32c.crc32c(string(window2),h)
    h = CRC32c.crc32c(string(window3), h)
    h = CRC32c.crc32c(string(window4),h)
    h = CRC32c.crc32c(string(nruns),h)
    if windowsize != 40.0
        h = CRC32c.crc32c(string(windowsize),h)
    end
    if !do_pca
        h = CRC32c.crc32c("no pace",h)
    end
    if use_locations != :all
        h = CRC32c.crc32c(string(use_locations),h)
    end
    if use_corners
        h = CRC32c.crc32c("use_corners",h)
    end
    qq = string(h, base=16)
    fname = joinpath("data","delay_response_orthogonal_subspace_$(qq).hdf5")

    if HDF5.ishdf5(fname) & !redo
        @show fname
        σ1, σ2, σ3, μ1, μ2, μ3, bins = HDF5.h5open(fname) do fid
            read(fid, "sigma1"), read(fid, "sigma2"), read(fid, "sigma3"), read(fid, "mu1"), read(fid,"mu2"), read(fid, "mu3"), read(fid, "bins")
        end
    else
        ppstht, labelst, rtimest = JLD2.load(joinpath("data","ppsth_fef_cue.jld2"), "ppsth", "labels","rtimes")
        X = ppstht.counts
        bins = ppstht.bins
        label = Vector{Vector{Int64}}(undef, length(labelst))
        X2 = fill(0.0, size(X,1), size(X,2), length(label))
        for ii in 1:length(label) 
            subject = DPHT.get_level_name("subject", ppstht.cellnames[ii])
            if use_corners
                lab = filter_corner_locations(subject, labelst[ii])
                tidx = findall(lab.!=nothing)
                label[ii] = lab[tidx]
                X2[:,1:length(tidx), ii] = X[:, tidx, ii]
            elseif use_locations != :all
                func = in(use_locations)
                tidx = findall(func, label[ii])
                label[i] = [findfirst(use_locations.==l) for l in label[ii][tidx]]
                X2[:,1:length(tidx), ii] = X[:, tidx, ii]
            else
                label[ii] = labelst[ii]
                X2[:, :, ii] .= X[:, :, ii]
            end
        end
        nc = maximum([maximum(l) for l in label])
        bins = bins[1:size(X,1)]

        RNG = MersenneTwister(1234)

        σ1 = fill(0.0, length(bins), nruns)
        σ2 = fill!(similar(σ1), 0.0)
        σ3 = fill!(similar(σ1), 0.0)
        σ4 = fill!(similar(σ1), 0.0)
        # hold the mean trajectories for each location for the decoder surfaces.
        μ1 = fill(0.0, nc-1, length(bins), nc, size(σ1,2))
        μ2 = fill(0.0, nc-1, length(bins), nc, size(σ1,2))
        μ3 = fill(0.0, nc-1, length(bins), nc, size(σ1,2))
        @showprogress 1.0 for jj in axes(σ1,2)
            Yt, train_label, test_label = EventOnsetDecoding.sample_trials(permutedims(X2, [3,2,1]), label; RNG=RNG, ntrain=1500, ntest=300)
            idx0 = searchsortedfirst(bins, window1[1])
            idx1 = searchsortedlast(bins, window1[2])
            Ytrain1 = permutedims(dropdims(mean(Yt[idx0:idx1,1:1500,:], dims=1),dims=1), [2,1])

            idx0 = searchsortedfirst(bins, window2[1])
            idx1 = searchsortedlast(bins, window2[2])
            Ytrain2 = permutedims(dropdims(mean(Yt[idx0:idx1,1:1500,:], dims=1),dims=1), [2,1])

            #go-cue
            idx0 = searchsortedfirst(bins, window3[1])
            idx1 = searchsortedlast(bins, window3[2])
            Ytrain3 = permutedims(dropdims(mean(Yt[idx0:idx1,1:1500,:], dims=1),dims=1), [2,1])

            idx0 = searchsortedfirst(bins, window4[1])
            idx1 = searchsortedlast(bins, window4[2])
            Ytrain4 = permutedims(dropdims(mean(Yt[idx0:idx1,1:1500,:], dims=1),dims=1), [2,1])

            #center
            for Y in [Ytrain1, Ytrain2]
                Y .-= mean(Y, dims=2)
            end
            if do_pca
                pca = fit(PCA, cat(Ytrain1, Ytrain2, Ytrain3, Ytrain4, dims=2), pratio=0.9)
                _Ytrain1 = predict(pca, Ytrain1)
                _Ytrain2 = predict(pca, Ytrain2)
                _Ytrain3 = predict(pca, Ytrain3)
                _Ytrain4 = predict(pca, Ytrain4)
            else
                _Ytrain1 = Ytrain1
                _Ytrain2 = Ytrain2
                _Ytrain3 = Ytrain3
                _Ytrain4 = Ytrain4
            end

            mstats1 = MultivariateStats.multiclass_lda_stats(Ytrain1, train_label)
            d1 = size(mstats1.Sb, 1)			

            mstats2 = MultivariateStats.multiclass_lda_stats(Ytrain2, train_label)
            d2 = size(mstats2.Sb, 1)

            mstats3 = MultivariateStats.multiclass_lda_stats(cat(_Ytrain3, _Ytrain4,dims=2), [fill(1, size(_Ytrain3,2));fill(2, size(_Ytrain4,2))])

            W = OrthogonalLDA.orthogonal_lda([mstats1.Sb, mstats2.Sb,mstats3.Sb], [mstats1.Sw,mstats2.Sw, mstats3.Sw], [nc-1,nc-1,1];debug=missing)

            W1 = W[:,1:nc-1]
            W2 = W[:, nc:end-1]
            W3 = W[:,end:end]
            #compute the projected means
            pmeans1 = fill(0.0, nc-1, nc)
            pmeans2 = fill(0.0, nc-1, nc)
            for l in 1:nc
                pmeans1[:,l] = mean(W1'*_Ytrain1[:,train_label.==l],dims=2)
                pmeans2[:,l] = mean(W2'*_Ytrain2[:,train_label.==l],dims=2)
            end
            pmeans3 = fill(0.0, 1, 2)
            pmeans3[:,1] = mean(W3'*_Ytrain3,dims=2)
            pmeans3[:,2] = mean(W3'*_Ytrain4,dims=2)

            _Yt,bins2 = rebin2(Yt[:,1501:end, :],bins, windowsize) 
            Ytest = permutedims(_Yt, [3,2,1])
            ntest = length(test_label)
            nlabel = fill(0, nc)
            for kk in axes(Ytest,3)
                fill!(nlabel, 0)
                for ll in axes(Ytest,2)
                    if do_pca
                        y = predict(pca, Ytest[:,ll,kk])
                    else
                        y = Ytest[:,ll,kk]
                    end
                    nlabel[test_label[ll]] += 1
                    for (σ, w, μ, pmeans) in zip([σ1, σ2], [W1, W2], [μ1, μ2], [pmeans1, pmeans2])
                        yp = w'*y
                        μ[:, kk,test_label[ll],jj] .+= yp
                        d = dropdims(sum(abs2, yp .- pmeans, dims=1),dims=1)
                        σ[kk,jj] += test_label[ll] == argmin(d)
                    end
                    yp = W3'*y
                    μ3[:, kk,test_label[ll],jj] .+= yp
                    d = dropdims(sum(abs2, yp .- pmeans3, dims=1),dims=1)
                    σ3[kk, jj] += argmin(d) == 2
                end
                σ1[kk,jj] /= ntest
                σ2[kk,jj] /= ntest
                σ3[kk,jj] /= ntest
                for ii in 1:nc
                    μ1[:,kk,ii,jj] ./= nlabel[ii]
                    μ2[:,kk,ii,jj] ./= nlabel[ii]
                    μ3[:,kk,ii,jj] ./= nlabel[ii]
                end
            end
        end
        HDF5.h5open(fname, "w") do fid
            fid["sigma1"] = σ1
            fid["sigma2"] = σ2
            fid["sigma3"] = σ3
            fid["mu1"] = μ1
            fid["mu2"] = μ2
            fid["mu3"] = μ3
            fid["bins"] = [bins;]
        end
    end
    σ1, σ2, σ3, μ1, μ2, μ3, bins
end

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
	# task for Whiske
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

"""
```
function plot_orthogonal_subspaces!(lg;windowsize=40.0, add_label=true, kvs...)
```
Plot the time resolved decoding performance of 3 orthognoal decoders
"""
function plot_orthogonal_subspaces!(lg;windowsize=40.0, add_label=true, kvs...)
    fontsize = get(kvs, :fontsize, 12)
    σ1, σ2, σ3, μ1, μ2, μ3, bins = find_orthogonal_subpspaces(; use_corners=true, redo=false, windowsize=windowsize)

    qqidx = 1:length(bins)-8
    Δb = windowsize/2.0 # half the window size of 40ms
    with_theme(plot_theme) do
        ax1 = Axis(lg[1,1],xlabelsize=fontsize, xticklabelsize=fontsize, ylabelsize=fontsize, yticklabelsize=fontsize)
        ax2 = Axis(lg[2,1],xlabelsize=fontsize, xticklabelsize=fontsize, ylabelsize=fontsize, yticklabelsize=fontsize)

        linkxaxes!(ax1, ax2)
        m1 = dropdims(mean(σ1[qqidx,:],dims=2),dims=2)
        s1 = dropdims(std(σ1[qqidx,:],dims=2),dims=2)
        m2 = dropdims(mean(σ2[qqidx,:],dims=2),dims=2)
        s2 = dropdims(std(σ2[qqidx,:],dims=2),dims=2)
        m3 = dropdims(mean(σ3[qqidx,:],dims=2),dims=2)
        s3 = dropdims(std(σ3[qqidx,:],dims=2),dims=2)
        binsq = bins[qqidx] .+ Δb
        band!(ax1, binsq, m1 - s1, m1 + s1)
        band!(ax1, binsq, m2 - s2, m2 + s2)
        band!(ax2, binsq, m3 - s3, m3 + s3, color=Cycled(3))
        lines!(ax1, binsq, m1, linewidth=2.0, label="Movement prep")
        lines!(ax1, binsq, m2, linewidth=2.0, label="Movement exc")
        lines!(ax2, binsq, m3, linewidth=2.0, color=Cycled(3), label="CI")
        hlines!(ax1, 1.0/size(μ1,3), linestyle=:dot, color="black")
        hlines!(ax2, 0.5, linestyle=:dot, color="black")
        for ax in [ax1, ax2]
            vlines!(ax, 0.0, color="black", linestyle=:dot)
        end
        axislegend(ax1, valign=:top, halign=:left, margin=(10.0, 0.0, 0.0, -17.0), padding=(10.0, 10.0, 10.0, 10.0))
        axislegend(ax2, valign=:top, halign=:left, margin=(10.0, 0.0, 0.0, -17.0), padding=(10.0, 10.0, 10.0, 10.0))

        ax1.ylabel = "Mov direction perf"
        ax2.ylabel = "Condition inv perf"
        ax2.xlabel = "Time from go-cue [ms]"
        ax1.xticklabelsvisible = false
        if add_label
            label_padding = (0.0, 0.0, 5.0, 0.0)
            
            labels = [Label(lg[1,1,TopLeft()], "A", padding=label_padding),
                    Label(lg[2,1,TopLeft()], "B", padding=label_padding)]
        end
    end
    lg
end

function plot(;do_save=true)
	#img = load(joinpath("figures","james_whiskey_implant_on_brain.png"))
    img = load(joinpath("figures","manuscript","electrodes_on_brain.png"))
	# grap the width and heigth from the QUARTO ENV variable
	width = parse(Float64, get(ENV, "QUARTO_FIG_WIDTH", "15"))*72
	height = parse(Float64, get(ENV, "QUARTO_FIG_HEIGHT","25"))*72
	with_theme(plot_theme) do
		fig = Figure(resolution=(width,height))
		lg1 = GridLayout()
		fig[1,1] = lg1
		task_figure!(lg1;fontsize=16)
		# TODO: Add the implant figure here. Just load it from a file
		lg2 = GridLayout()
		fig[2,1] = lg2
		ax = Axis(lg2[1,1], xticksvisible=false, yticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false,leftspinevisible=false, bottomspinevisible=false,aspect=DataAspect())
		image!(ax,rotr90(img))
		rowsize!(fig.layout, 2, 442)

		lg3 = GridLayout()
		fig[3,1] = lg3
		plot_orthogonal_subspaces!(lg3;add_label=false, fontsize=16)
		#add labels
		label_fontsize = 24 
		labels = [Label(lg1[1,1,TopLeft()], "A",fontsize=label_fontsize),
				  Label(lg2[1,1,TopLeft()], "B",fontsize=label_fontsize),
				  Label(lg3[1,1,TopLeft()], "C",fontsize=label_fontsize)]
		fname = joinpath("figures","manuscript","task_implants_and_spaces.pdf")
        if do_save
            save(fname, fig;pt_per_unit=1)
        end
		fig
	end
end