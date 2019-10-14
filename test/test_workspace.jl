function init_setup()
    param = :complexConjug
    θ₀ = [10.0, -5.0, 5.0, 0.0, 3.0]
    P˟ = BSI.FitzhughDiffusion(param, θ₀...)
    obs = [[0.0, 0.0], [-0.3, -0.2], [0.4, 0.2]]
    tt = [0.0, 1.0, 1.5]
    P̃ = [BSI.FitzhughDiffusionAux(param, θ₀..., tt[1], obs[1], tt[2], obs[2]),
         BSI.FitzhughDiffusionAux(param, θ₀..., tt[2], obs[2], tt[3], obs[3])]
    setup = BSI.MCMCSetup(P˟, P̃, BSI.PartObs())
    (setup = setup, θ = θ₀, trgt = P˟, obs = obs, tt = tt, aux = P̃)
end

@testset "acceptance tracker" begin
    setup = init_setup().setup
    updt_coord = ((1,2),(2,3))
    BSI.set_transition_kernels!(setup, nothing, nothing, true, updt_coord,
                            (BSI.MetropolisHastingsUpdt(),BSI.MetropolisHastingsUpdt()))
    at = BSI.AccptTracker(setup)

    @testset "initialisation" begin
        @test at.accpt_imp == 0
        @test at.prop_imp == 0
        @test at.accpt_updt == [0, 0]
        @test at.prop_updt == [0, 0]
        @test at.updt_len == length(updt_coord)
    end

    accept_reject = [true, false, false, true, true, false]

    for ar in accept_reject
        BSI.update!(at, BSI.ParamUpdate(), 1, ar)
        BSI.update!(at, BSI.Imputation(), ar)
    end

    @testset "testing update! (1/2)" begin
        accept_rate = sum(accept_reject)/length(accept_reject)
        @test BSI.accpt_rate(at, BSI.ParamUpdate())[1] == accept_rate
        @test isnan(BSI.accpt_rate(at, BSI.ParamUpdate())[2])
        @test BSI.accpt_rate(at, BSI.Imputation()) == accept_rate
    end

    for ar in accept_reject BSI.update!(at, BSI.ParamUpdate(), 2, ar) end

    @testset "testing update! (2/2)" begin
        accept_rate = sum(accept_reject)/length(accept_reject)
        @test BSI.accpt_rate(at, BSI.ParamUpdate())[1] == accept_rate
        @test BSI.accpt_rate(at, BSI.ParamUpdate())[2] == accept_rate
        @test BSI.accpt_rate(at, BSI.Imputation()) == accept_rate
    end

end


@testset "parameter history" begin
    out = init_setup()
    setup, θ = out.setup, out.θ
    updt_coord = ((1,2),(2,3))
    BSI.set_transition_kernels!(setup, nothing, nothing, true, updt_coord,
                            (BSI.MetropolisHastingsUpdt(),BSI.MetropolisHastingsUpdt()))

    num_mcmc_steps = 1000
    warm_up = 50
    BSI.set_mcmc_params!(setup, num_mcmc_steps, nothing, nothing, nothing, warm_up)
    ph = BSI.ParamHistory(setup)

    _foo(x::BSI.ParamHistory{T}) where T = T

    @testset "initialisation" begin
        @test eltype(ph.θ_chain) == typeof(θ) == _foo(ph)
        @test length(ph.θ_chain) == length(updt_coord)*(num_mcmc_steps-warm_up)+1
        @test ph.θ_chain[1] == last(ph) == θ
        @test ph.counter == 1
    end

    N = 10
    chain = [rand(5) for i in 1:N]
    for i in 1:N BSI.update!(ph, chain[i]) end

    @testset "updating chain" begin
        @test ph.θ_chain[1] == θ
        @test all([ph.θ_chain[i+1] == chain[i] for i in 1:N])
        @test BSI.last(ph) == ph.θ_chain[N+1] == chain[N]
        @test BSI.last(ph) != θ
    end
end

@testset "action tracker" begin
    setup = init_setup().setup
    save_iter = 10
    verb_iter = 3
    warm_up = 50
    BSI.set_mcmc_params!(setup, nothing, save_iter, verb_iter, nothing, warm_up)
    param_updt = true
    BSI.set_transition_kernels!(setup, nothing, nothing, param_updt)
    at = BSI.ActionTracker(setup)

    @testset "initialisation" begin
        @test at.save_iter == save_iter
        @test at.verb_iter == verb_iter
        @test at.warm_up == warm_up
        @test at.param_updt == param_updt
    end

    @testset "correct acting" begin
        @test !BSI.act(BSI.SavePath(), at, 1)
        @test !BSI.act(BSI.SavePath(), at, 10)
        @test BSI.act(BSI.SavePath(), at, 60)
        @test !BSI.act(BSI.SavePath(), at, 51)
        @test !BSI.act(BSI.SavePath(), at, 61)
        @test BSI.act(BSI.SavePath(), at, 40000)

        @test !BSI.act(BSI.Verbose(), at, 1)
        @test BSI.act(BSI.Verbose(), at, 3)
        @test !BSI.act(BSI.Verbose(), at, 10)
        @test BSI.act(BSI.Verbose(), at, 30)

        @test !BSI.act(BSI.ParamUpdate(), at, 1)
        @test !BSI.act(BSI.ParamUpdate(), at, 10)
        @test !BSI.act(BSI.ParamUpdate(), at, 50)
        @test BSI.act(BSI.ParamUpdate(), at, 51)
        @test BSI.act(BSI.ParamUpdate(), at, 1000)
    end
end


@testset "workspace" begin
    out = init_setup()
    setup, θ, trgt, obs, tt = out.setup, out.θ, out.trgt, out.obs, out.tt
    aux = out.aux

    L = [1. 0.; 0. 1.]
    Σ = [0.5 0.0; 0.0 1.0]
    BSI.set_observations!(setup, [L, L], [Σ, 2*Σ], obs, tt)

    dt = 0.01
    τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
    BSI.set_imputation_grid!(setup, dt, τ)

    t_kernels = [BSI.RandomWalk([0.002, 0.1], [true, true]),
                 BSI.RandomWalk([0.2, 1.0], [false, true])]
    ρ = 0.5
    param_updt = true
    updt_coord = (Val((true,true,false)),
                  Val((false,true,true)))
    updt_type=(BSI.MetropolisHastingsUpdt(),
               BSI.ConjugateUpdt())
    BSI.set_transition_kernels!(setup, t_kernels, ρ, param_updt, updt_coord,
                            updt_type)

    priors = BSI.Priors((BSI.ImproperPrior(), BSI.ImproperPrior()))
    x0_prior = BSI.KnownStartingPt(obs[1])
    BSI.set_priors!(setup, priors, x0_prior)

    num_mcmc_steps = 100
    BSI.set_mcmc_params!(setup, num_mcmc_steps)

    BSI.set_solver!(setup)

    BSI.initialise!(Float64, setup)

    ws = BSI.Workspace(setup).workspace
    @testset "initialisation" begin
        @test typeof(ws.Wnr) == Wiener{Float64}
        @test length(ws.XX) == length(ws.XXᵒ) == 2
        @test length(ws.WWᵒ) == length(ws.WW) == 2
        @test all([ws.XX[i].tt == ws.XXᵒ[i].tt for i in 1:2])
        @test all([ws.XX[i].yy == ws.XXᵒ[i].yy for i in 1:2])
        @test all([ws.WW[i].tt == ws.WWᵒ[i].tt for i in 1:2])
        @test all([ws.WW[i].yy == ws.WWᵒ[i].yy for i in 1:2])
        @test eltype(eltype(ws.WW)) == Float64
        @test eltype(eltype(ws.XX)) == SArray{Tuple{2},Float64,1,2}
        @test ws.WW[1].tt[1] == ws.XX[1].tt[1] == tt[1]
        @test ws.WW[2].tt[1] == ws.XX[2].tt[1] == tt[2]
        @test typeof(ws.P[1].Target) <: BSI.FitzhughDiffusion
        @test typeof(ws.P[1].Pt) <: BSI.FitzhughDiffusionAux
        @test all([ws.P[i].tt == ws.WW[i].tt == ws.XX[i].tt for i in 1:2])
        @test ws.recompute_ODEs == [true, true]
        @test ws.ρ == 0.5
    end

    ws2 = Workspace(ws, 0.25)
    @testset "copy constructor" begin
        @test ws2.ρ == 0.25
        @test ws.ρ == 0.5
        @test ws.Wnr == ws2.Wnr
        @test ws.XX == ws2.XX
        @test ws.WW == ws2.WW
        @test ws.P == ws.P
    end

    @testset "action determination" begin
        @test !BSI.act(BSI.SavePath(), ws, 1)
        @test !BSI.act(BSI.SavePath(), ws, 10)
        @test !BSI.act(BSI.SavePath(), ws, 60)
        @test !BSI.act(BSI.SavePath(), ws, 51)
        @test !BSI.act(BSI.SavePath(), ws, 61)
        @test !BSI.act(BSI.SavePath(), ws, 40000)

        @test !BSI.act(BSI.Verbose(), ws, 1)
        @test !BSI.act(BSI.Verbose(), ws, 3)
        @test !BSI.act(BSI.Verbose(), ws, 10)
        @test !BSI.act(BSI.Verbose(), ws, 30)

        @test BSI.act(BSI.ParamUpdate(), ws, 1)
        @test BSI.act(BSI.ParamUpdate(), ws, 10)
        @test BSI.act(BSI.ParamUpdate(), ws, 50)
        @test BSI.act(BSI.ParamUpdate(), ws, 51)
        @test BSI.act(BSI.ParamUpdate(), ws, 1000)
    end
end



@testset "definition of gibbs sweep" begin
    out = init_setup()
    setup, obs, tt = out.setup, out.obs, out.tt
    L = [1. 0.; 0. 1.]
    Σ = [0.5 0.0; 0.0 1.0]
    BSI.set_observations!(setup, [L, L], [Σ, 2*Σ], obs, tt)

    dt = 0.01
    τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
    BSI.set_imputation_grid!(setup, dt, τ)

    t_kernels = [BSI.RandomWalk([0.002, 0.1], [true, true]),
                 BSI.RandomWalk([0.2, 1.0], [false, true])]
    ρ = 0.5
    param_updt = true
    updt_coord = (Val((true,true,false)),
                  Val((false,true,true)))
    updt_type=(BSI.MetropolisHastingsUpdt(),
               BSI.ConjugateUpdt())
    BSI.set_transition_kernels!(setup, t_kernels, ρ, param_updt, updt_coord,
                            updt_type)

    priors = BSI.Priors((BSI.ImproperPrior(), BSI.ImproperPrior()))
    x0_prior = BSI.KnownStartingPt(obs[1])
    BSI.set_priors!(setup, priors, x0_prior)

    num_mcmc_steps = 100
    BSI.set_mcmc_params!(setup, num_mcmc_steps)

    BSI.set_solver!(setup)

    BSI.initialise!(Float64, setup)
    gs = BSI.GibbsDefn(setup)
    @testset "initialisation" begin
        @test length(gs) == 2
        @test all([gs[i] == gs.updates[i] for i in 1:2])
        @test all([gs[i].updt_type == updt_type[i] for i in 1:2])
        @test all([gs[i].updt_coord == updt_coord[i] for i in 1:2])
        @test all([gs[i].priors == priors[i] for i in 1:2])
        @test gs[1].recompute_ODEs && gs[2].recompute_ODEs
    end
end
