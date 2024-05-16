module FigureS2
using ..Figure2
using ..PlotUtils

using CairoMakie

function plot(;do_savel=false)
    plot_single_cell_examples([("W", 45), ("W", 54), ("W", 8),("W",93),("W",51), ("J", 92)], 35.0;height=1300,labels=["A","B","C","D","E","F"], toprowsize=Relative(0.4))
end

function plot_single_cell_examples(cells::Vector{Tuple{String, Int64}}, windowsize;labels::Vector{String}=["A","B","C"], width=691, height=950,
                                        tmin=(target=-100.0, cue=-100.0, mov=-300.0),
                                        tmax=(target=500.0, cue=500.0, mov=100.0), kvs...)
    toprowsize = Relative(0.3) 
    with_theme(plot_theme) do
        fig = Figure(resolution=(width,height))

        # visual cell
        lg11 = GridLayout()
        fig[1,1] = lg11
        Figure2.plot_psth_and_raster!(lg11, cells[1][1], cells[1][2],windowsize;ylabelvisible=false, xlabelvisible=false, xticklabelsvisible=false, toprowsize=toprowsize, tmin=tmin, tmax=tmax, kvs...)

        # movement cell
        lg21 = GridLayout()
        fig[2,1] = lg21
        Figure2.plot_psth_and_raster!(lg21, cells[2][1], cells[2][2],windowsize;ylabelvisible=false, xlabelvisible=false, xticklabelsvisible=false, toprowsize=toprowsize, tmin=tmin, tmax=tmax, kvs...)

        # visuo-movement cell
        lg31 = GridLayout()
        fig[3,1] = lg31
        Figure2.plot_psth_and_raster!(lg31, cells[3][1], cells[3][2],windowsize; ylabelvisible=false, xlabelvisible=false, xticklabelsvisible=false, toprowsize=toprowsize, tmin=tmin, tmax=tmax, kvs...)

        lg41 = GridLayout()
        fig[4,1] = lg41
        Figure2.plot_psth_and_raster!(lg41, cells[4][1], cells[4][2],windowsize;ylabelvisible=false, xlabelvisible=false,  xticklabelsvisible=false, toprowsize=toprowsize, tmin=tmin, tmax=tmax, kvs...)

        lg51 = GridLayout()
        fig[5,1] = lg51
        Figure2.plot_psth_and_raster!(lg51, cells[5][1], cells[5][2],windowsize; ylabelvisible=false, xlabelvisible=false, xticklabelsvisible=false, toprowsize=toprowsize, tmin=tmin, tmax=tmax, kvs...)

        lg61 = GridLayout()
        fig[6,1] = lg61
        Figure2.plot_psth_and_raster!(lg61, cells[6][1], cells[6][2],windowsize; xlabelvisible=true,toprowsize=toprowsize, tmin=tmin, tmax=tmax, kvs...)

        lmargin = (20,20,10,10)
        labels = [Label(fig[i,1, TopLeft()], label, padding=lmargin) for (i,label) in enumerate(labels)]
        rowsize!(fig.layout, 6, Relative(1.5/6))
        for i in 1:5
            rowgap!(fig.layout, i, 0.5)
        end
    fig
    end
end
end #module