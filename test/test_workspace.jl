function init_setup()
    param = :complexConjug
    θ₀ = [10.0, -5.0, 5.0, 0.0, 3.0]
    P˟ = BSI.FitzhughDiffusion(param, θ₀...)
    obs = [[0.0, 0.0], [-0.3, -0.2], [0.4, 0.2]]
    tt = [0.0, 1.0, 1.5]
    P̃ = [BSI.FitzhughDiffusionAux(param, θ₀..., tt[1], obs[1], tt[2], obs[2]),
         BSI.FitzhughDiffusionAux(param, θ₀..., tt[2], obs[2], tt[3], obs[3])]
    setup = BSI.DiffusionSetup(P˟, P̃, BSI.PartObs())
    (setup = setup, θ = θ₀, trgt = P˟, obs = obs, tt = tt, aux = P̃)
end


@testset "mcmc schedule" begin
    num_mcmc_steps = 20
    updt_idx = [[1,2,3],[4,5],[6]]
    actions = (save=10, verbose=3, warm_up=5,
               readjust=(x->x%7==0), fuse=(x->false))
    schedule = BSI.MCMCSchedule(num_mcmc_steps, updt_idx, actions)

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

    x0_prior = BSI.KnownStartingPt(obs[1])
    BSI.set_x0_prior!(setup, x0_prior)

    BSI.initialise!(Float64, setup, Vern7(), false, NoChangePt())

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
    end
end
