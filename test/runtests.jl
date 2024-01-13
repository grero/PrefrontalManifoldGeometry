using Test
using PrefrontalManifoldGeometry
const PMG = PrefrontalManifoldGeometry

@testset "Figures" begin
    fig1 = PMG.Figure1().plot()
    fig2 = PMG.Figure2().plot()
    fig3 = PMG.Figure3().plot()
    fig4 = PMG.Figure4().plot()
    fig5 = PMG.Figure5().plot()
end

