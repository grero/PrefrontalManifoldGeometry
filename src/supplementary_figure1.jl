using Makie
using Colors
using CairoMakie

using JLD2


function plot(;do_save=true, kvs...)
    fnamec = joinpath("data","ppsth_fef_cue_raw.jld2")
    rtimes = JLD2.load(fnamec, "rtimes")
    rtimes_subject = Dict{String,Vector{Vector{Float64}}}()
    for subject in ["J","W"]
        rtimes_subject[subject] = Vector{Float64}[]
        sessions = filter(contains(subject), keys(rtimes))
        for session in sessions
            rtime = filter(rt-> 90.0 .< rt <= 300.0, rtimes[session])
            push!(rtimes_subject[subject], rtime)
        end
    end

    with_theme(plot_theme) do
		width = 5*72
		height = 4/3*width
		fig = Figure(resolution=(width,height))
		ax = Axis(fig[1,1])
		ax2 = Axis(fig[2,1])
		linkxaxes!(ax, ax2)
		cc1 = LCHuv(ax.palette.color[][1])
		cc2 = LCHuv(ax.palette.color[][2])
		colors1 = [LCHuv(cc1.l, a*cc1.c, cc1.h) for a in range(0.5, stop=1.8,length=length(rtimes_subject["W"]))]

		colors2 = [LCHuv(cc2.l, a*cc2.c, cc2.h) for a in range(0.5, stop=1.8,length=length(rtimes_subject["J"]))]
		offset = 0
		for (subject,colors) in zip(["W","J"],[colors1, colors2])
			for (color,rtime) in zip(colors, rtimes_subject[subject])
				dd = StatsBase.fit(Gamma, rtime)
				x = sort(rtime)
				y = pdf.(dd,x)
				y ./= sum(y)
				#hh = StatsBase.fit(Histogram, rtime;nbins=20)
				#stairs!(ax, hh.edges[1][1:end-1], hh.weights)
				lines!(ax, x, y,color=color,label=subject)
				#scatter!(ax, x, y, color=color,markersize=5px)
				#hist!(ax, rtime,color=color)
				scatter!(ax2, rtime, offset .+ 0.7*rand(length(rtime)), markersize=2.5px, color=color)
				offset += 1
			end
		end
		group_color1 = [PolyElement(color = color, strokecolor = :transparent) for color in colors1] 
		group_color2 = [PolyElement(color = color, strokecolor = :transparent) for color in colors2]
		legend = Legend(fig[1:2,2], [group_color1, group_color2], [["" for i in 1:length(colors1)],["" for i in 1:length(colors2)]], ["Monkey W", "Monkey J"],tellwidth=false, valign=:top, halign=:right)
		ax.ylabel = "PDF"
		ax.xticklabelsvisible = false
		ax2.xlabel = "Reaction time [ms]"
        ax2.xticks = [100.0, 200.0, 300.0]
		ax2.yticklabelsvisible = false
		ax2.yticksvisible = false
		ax2.leftspinevisible = false
		ax.yticklabelsvisible = true
		rowsize!(fig.layout, 1, Relative(0.6))
        if do_save
            fname = joinpath("figures","manuscript","reaction_time_distributions.pdf")
            save(fname, fig;pt_per_unit=1)
        end
		fig
	end
end