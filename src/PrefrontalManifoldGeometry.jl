module PrefrontalManifoldGeometry
using StatsBase

include("utils.jl")
include("plot_utils.jl")
include("trajectories.jl")
include("regression.jl")
include("csi_analysis.jl")

using .Utils
using .PlotUtils

include("figure1.jl")
include("figure2.jl")
include("figure3.jl")
include("figure4.jl")
include("figure5.jl")

include("target_and_go_cue_comparsion.jl")

# supplementaru
include("supplementary_figure1.jl")
include("supplementary_figure2.jl")
include("supplementary_figure3.jl")
include("supplementary_figure5.jl")
include("supplementary_figure6.jl")
include("supplementary_figure7.jl")
include("supplementary_figure8.jl")
include("supplementary_figure12.jl")
end
