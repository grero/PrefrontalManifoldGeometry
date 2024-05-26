using Test
using TestItems
using PrefrontalManifoldGeometry
using StableRNGs
const PMG = PrefrontalManifoldGeometry

@testset "Regression" begin
    @testitem "Regression" begin
        rng = StableRNG(1234)
        xt = randn(rng, 300,1)
        y = 0.1.*xt[:,1] .+ 0.3
        z = randn(rng, 300,1)
        β, Δβ, pv, r², rss, varidx = PrefrontalManifoldGeometry.Figure3.compute_regression(lrt, xt, z;use_residuals=true, save_all_β=true)
        @show pv
    end
end

@testset "Figures" begin
    fig1 = PMG.Figure1().plot()
    fig2 = PMG.Figure2().plot()
    fig3 = PMG.Figure3().plot()
    fig4 = PMG.Figure4().plot()
    fig5 = PMG.Figure5().plot()
end

