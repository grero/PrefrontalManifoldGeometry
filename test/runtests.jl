using Test
using TestItems
using PrefrontalManifoldGeometry
using StableRNGs
const PMG = PrefrontalManifoldGeometry

@testset "Responsive and selective" begin
    nt = 20
    nbins = 3
    X = fill(0.0, nbins, nt)
    label = rand(1:2, nt)
    n1 = sum(label.==1)
    n2 = sum(label.==2)
    X[1,:] .= 0.1*randn(nt)
    X[2,label.==1] .= 1.0 .+ 0.1*randn(n1)
    X[2,label.==2] .= 0.5 .+ 0.1*randn(n2)
    X[3,label.==1] .= 0.5 .+ 0.1*randn(n1)
    X[3,label.==2] .= 1.0 .+ 0.1*randn(n2)
    is_responsive, is_selective = PMG.Figure2.classify_cell(X, [0,1,2], label, [(0,1),(1,2), (2,3)])
    @test is_responsive == [true, true]
    @test is_selective == [true, true]
end
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
    fig1 = PMG.Figure1.plot()
    fig2 = PMG.Figure2.plot()
    fig3 = PMG.Figure3.plot()
    fig4 = PMG.Figure4.plot()
    fig5 = PMG.Figure5.plot()
end

@testset "Supplementary Figures" begin
    figs1 = PMG.FigureS1.plot()
    figs2 = PMG.FigureS2.plot()
    figs3 = PMG.FigureS3.plot()

    figs5 = PMG.FigureS5.plot()
    figs6 = PMG.FigureS6.plot()
    figs7 = PMG.FigureS7.plot()
    figs8 = PMG.FigureS8.plot()
end

