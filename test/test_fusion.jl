@testset "fusion" begin
    θ = fill(1.0, 5)
    setup = MCMCSetup(
        BSI.Imputation(BSI.NoBlocking(), 0.5, BSI.Vern7()),
        BSI.Imputation(BSI.NoBlocking(), 0.5, BSI.Vern7()),
        BSI.ParamUpdate(BSI.MetropolisHastingsUpdt(), [1], θ,
                        BSI.UniformRandomWalk(0.5, true),
                        BSI.ImproperPosPrior(),
                        BSI.UpdtAuxiliary(BSI.Vern7(), true)),
        BSI.ParamUpdate(BSI.MetropolisHastingsUpdt(), [2], θ,
                        BSI.UniformRandomWalk(0.5, true),
                        BSI.ImproperPosPrior(),
                        BSI.UpdtAuxiliary(BSI.Vern7(), false)),
        BSI.ParamUpdate(BSI.MetropolisHastingsUpdt(), [4], θ,
                        BSI.UniformRandomWalk(0.5, true),
                        BSI.ImproperPosPrior(),
                        BSI.UpdtAuxiliary(Vern7(), false)))
    updt_idx_af = [[1,6],[2,6]]
    schedule = BSI.MCMCSchedule(19, [[1,3],[2,4,5]],
                                (save=5, verbose=4, warm_up=3,
                                readjust=(x->false), fuse=(x->x%10==0)))
    ws = BSI.MCMCWorkspace(setup, schedule, θ)
    cov_mat = rand(5,5)
    ws.cov .= copy(cov_mat)
    for step in schedule
        step.fuse && BSI.fuse!(ws, schedule)
        if step.iter >= 10
            @test step.idx == updt_idx_af[mod1(step.iter, 2)]
        end
    end
    @test length(ws.updates) == 6
    foo = ws.updates[6]
    @test foo.updt_coord == Val{(true, true, false, true, false)}()
    @test foo.t_kernel.gsn_A.Σ == 2.38^2/3.0*Matrix(view(cov_mat, [1,2,4], [1,2,4]))
    @test foo.t_kernel.gsn_B.Σ == 0.1^2/3.0*Matrix{Float64}(I, 3, 3)
    @test all(foo.t_kernel.gsn_A.pos)
    @test all(foo.t_kernel.gsn_B.pos)
    @test foo.priors.priors == (BSI.ImproperPosPrior(), BSI.ImproperPosPrior(),
                               BSI.ImproperPosPrior())
    @test foo.priors.coord_idx == (Val{(true, false, false, false, false)}(),
                                  Val{(false, true, false, false, false)}(),
                                  Val{(false, false, false, true, false)}())
end
