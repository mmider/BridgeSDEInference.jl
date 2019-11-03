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
    reset!(at)
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
    reset!(at_vec)
    @testset "vector after reset" begin
        @test all([a.prop == 0 for a in at_vec])
        @test all([a.accpt == 0 for a in at_vec])
    end
end

@testset "mcmc schedule" begin
    num_mcmc_steps = 20
    updt_idx = [[1,2,3],[4,5],[6]]
    actions = (save=10, verbose=3, warm_up=5,
               readjust=(x->x%7==0), fuse=(x->false))
    schedule = MCMCSchedule(num_mcmc_steps, updt_idx, actions)

    @testset "initialisation" begin
        @test schedule.num_mcmc_steps == num_mcmc_steps
        @test schedule.updt_idx == updt_idx
        @test schedule.actions == actions
    end

    @testset "correct iterations" begin
        i = 0
        for step in schedule
            i += 1
            @test step.iter == i
            @test step.idx == updt_idx[mod1(i, length(updt_idx))]
            @test step.save == (i in [10, 20])
            @test step.verbose == (i % actions.verbose == 0)
            @test step.param_updt == (i > actions.warm_up)
            @test step.readjust == (i % 7 == 0)
            @test step.fuse == false
        end
        @test i == 20
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
        @test ws.ρ == [[0.5]]
    end

    # NOTE this constructor has been deprecated
    #ws2 = Workspace(ws, 0.25)
    #@testset "copy constructor" begin
    #    @test ws2.ρ == 0.25
    #    @test ws.ρ == 0.5
    #    @test ws.Wnr == ws2.Wnr
    #    @test ws.XX == ws2.XX
    #    @test ws.WW == ws2.WW
    #    @test ws.P == ws.P
    #end

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
