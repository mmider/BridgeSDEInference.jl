@testset "acceptance tracker" begin
    at = BSI.AccptTracker(0)

    @testset "initialisation" begin
        @test at.accpt == 0
        @test at.prop == 0
    end

    accept_reject = [true, false, false, true, true, false]
    for ar in accept_reject BSI.register_accpt!(at, ar) end

    @testset "testing update!" begin
        @test BSI.acceptance_rate(at) == sum(accept_reject)/length(accept_reject)
    end
    BSI.reset!(at)
    @testset "after reset" begin
        @test at.accpt == 0
        @test at.prop == 0
    end

    at_vec = [BSI.AccptTracker(0) for i in 1:6]
    BSI.register_accpt!(at_vec, accept_reject)
    @testset "testing registration for vector of accpt tracker!" begin
        @test all([a.prop == 1 for a in at_vec])
        @test all([at_vec[i].accpt == 1*accept_reject[i] for i in 1:length(at_vec)])
    end
    BSI.reset!(at_vec)
    @testset "vector after reset" begin
        @test all([a.prop == 0 for a in at_vec])
        @test all([a.accpt == 0 for a in at_vec])
    end
end


@testset "definition of parameter update" begin
    θ = fill(0.0,5) # just for the dimension
    pu1 = BSI.ParamUpdate(BSI.MetropolisHastingsUpdt(), (1,), θ,
                          BSI.UniformRandomWalk(1.0,true), BSI.ImproperPosPrior(),
                          BSI.UpdtAuxiliary(BSI.Vern7(), true))
    mvn_prior = MvNormal([0.0,0.0], diagm(0=>fill(1000.0, 2)))
    pu2 = BSI.ParamUpdate(BSI.ConjugateUpdt(), (2,3), θ, nothing, mvn_prior,
                          BSI.UpdtAuxiliary(BSI.Vern7(), true))
    @testset "initialisation" begin
        @test typeof(pu1) <: BSI.ParamUpdate{BSI.MetropolisHastingsUpdt}
        @test typeof(pu2) <: BSI.ParamUpdate{BSI.ConjugateUpdt}
        @test pu1.updt_coord == Val((true,false,false,false,false))
        @test pu2.updt_coord == Val((false,true,true,false,false))
        @test pu1.t_kernel.ϵ == 1.0
        @test pu1.t_kernel.pos
        @test pu2.t_kernel === nothing
        @test typeof(pu1.priors) <: BSI.Priors
        @test typeof(pu2.priors) <: BSI.Priors
        @test typeof(pu1.priors.priors) <: Tuple{BSI.ImproperPosPrior}
        @test pu2.priors.priors == (mvn_prior,)
    end
end
