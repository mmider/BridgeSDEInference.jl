include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "mcmc", "setup.jl"))
include(joinpath(SRC_DIR, "mcmc", "workspace.jl"))
include(joinpath(SRC_DIR, "examples", "fitzHughNagumo.jl"))
include(joinpath(SRC_DIR, "transition_kernels", "random_walk.jl"))
include(joinpath(SRC_DIR, "mcmc_extras", "adaptation.jl"))
include(joinpath(SRC_DIR, "mcmc", "priors.jl"))
include(joinpath(SRC_DIR, "stochastic_process", "guid_prop_bridge.jl"))
include(joinpath(SRC_DIR, "mcmc_extras", "starting_pt.jl"))
include(joinpath(SRC_DIR, "solvers", "ralston3.jl"))
include(joinpath(SRC_DIR, "stochastic_process", "bounded_diffusion_domain.jl"))
include(joinpath(SRC_DIR, "solvers", "euler_maruyama_dom_restr.jl"))
include(joinpath(SRC_DIR, "mcmc_extras", "first_passage_times.jl"))
include(joinpath(SRC_DIR, "mcmc_extras", "blocking_schedule.jl"))
include(joinpath(SRC_DIR, "mcmc", "mcmc.jl"))

function init_setup()
    param = :complexConjug
    θ₀ = [10.0, -5.0, 5.0, 0.0, 3.0]
    P˟ = FitzhughDiffusion(param, θ₀...)
    obs = [[0.0, 0.0], [-0.3, -0.2], [0.4, 0.2]]
    tt = [0.0, 1.0, 1.5]
    P̃ = [FitzhughDiffusionAux(param, θ₀..., tt[1], obs[1], tt[2], obs[2]),
         FitzhughDiffusionAux(param, θ₀..., tt[2], obs[2], tt[3], obs[3])]
    setup = MCMCSetup(P˟, P̃, PartObs())
    (setup = setup, θ = θ₀, trgt = P˟, obs = obs, tt = tt, aux = P̃)
end

@testset "acceptance tracker" begin
    setup = init_setup().setup
    updt_coord = ((1,2),(2,3))
    set_transition_kernels!(setup, nothing, nothing, true, updt_coord,
                            (MetropolisHastingsUpdt(),MetropolisHastingsUpdt()))
    at = AccptTracker(setup)

    @testset "initialisation" begin
        @test at.accpt_imp == 0
        @test at.prop_imp == 0
        @test at.accpt_updt == [0, 0]
        @test at.prop_updt == [0, 0]
        @test at.updt_len == length(updt_coord)
    end

    accept_reject = [true, false, false, true, true, false]

    for ar in accept_reject
        update!(at, ParamUpdate(), 1, ar)
        update!(at, Imputation(), ar)
    end

    @testset "testing update! (1/2)" begin
        accept_rate = sum(accept_reject)/length(accept_reject)
        @test accpt_rate(at, ParamUpdate())[1] == accept_rate
        @test isnan(accpt_rate(at, ParamUpdate())[2])
        @test accpt_rate(at, Imputation()) == accept_rate
    end

    for ar in accept_reject update!(at, ParamUpdate(), 2, ar) end

    @testset "testing update! (2/2)" begin
        accept_rate = sum(accept_reject)/length(accept_reject)
        @test accpt_rate(at, ParamUpdate())[1] == accept_rate
        @test accpt_rate(at, ParamUpdate())[2] == accept_rate
        @test accpt_rate(at, Imputation()) == accept_rate
    end

end


@testset "parameter history" begin
    out = init_setup()
    setup, θ = out.setup, out.θ
    updt_coord = ((1,2),(2,3))
    set_transition_kernels!(setup, nothing, nothing, true, updt_coord,
                            (MetropolisHastingsUpdt(),MetropolisHastingsUpdt()))

    num_mcmc_steps = 1000
    warm_up = 50
    set_mcmc_params!(setup, num_mcmc_steps, nothing, nothing, nothing, warm_up)
    ph = ParamHistory(setup)

    _foo(x::ParamHistory{T}) where T = T

    @testset "initialisation" begin
        @test eltype(ph.θ_chain) == typeof(θ) == _foo(ph)
        @test length(ph.θ_chain) == length(updt_coord)*(num_mcmc_steps-warm_up)+1
        @test ph.θ_chain[1] == last(ph) == θ
        @test ph.counter == 1
    end

    N = 10
    chain = [rand(5) for i in 1:N]
    for i in 1:N update!(ph, chain[i]) end

    @testset "updating chain" begin
        @test ph.θ_chain[1] == θ
        @test all([ph.θ_chain[i+1] == chain[i] for i in 1:N])
        @test last(ph) == ph.θ_chain[N+1] == chain[N]
        @test last(ph) != θ
    end
end

@testset "action tracker" begin
    setup = init_setup().setup
    save_iter = 10
    verb_iter = 3
    warm_up = 50
    set_mcmc_params!(setup, nothing, save_iter, verb_iter, nothing, warm_up)
    param_updt = true
    set_transition_kernels!(setup, nothing, nothing, param_updt)
    at = ActionTracker(setup)

    @testset "initialisation" begin
        @test at.save_iter == save_iter
        @test at.verb_iter == verb_iter
        @test at.warm_up == warm_up
        @test at.param_updt == param_updt
    end

    @testset "correct acting" begin
        @test !act(SavePath(), at, 1)
        @test !act(SavePath(), at, 10)
        @test act(SavePath(), at, 60)
        @test !act(SavePath(), at, 51)
        @test !act(SavePath(), at, 61)
        @test act(SavePath(), at, 40000)

        @test !act(Verbose(), at, 1)
        @test act(Verbose(), at, 3)
        @test !act(Verbose(), at, 10)
        @test act(Verbose(), at, 30)

        @test !act(ParamUpdate(), at, 1)
        @test !act(ParamUpdate(), at, 10)
        @test !act(ParamUpdate(), at, 50)
        @test act(ParamUpdate(), at, 51)
        @test act(ParamUpdate(), at, 1000)
    end
end


@testset "workspace" begin
    out = init_setup()
    setup, θ, trgt, obs, tt = out.setup, out.θ, out.trgt, out.obs, out.tt
    aux = out.aux

    L = [1. 0.; 0. 1.]
    Σ = [0.5 0.0; 0.0 1.0]
    set_observations!(setup, [L, L], [Σ, 2*Σ], obs, tt)

    dt = 0.01
    τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
    set_imputation_grid!(setup, dt, τ)

    t_kernels = [RandomWalk([0.002, 0.1], [true, true]),
                 RandomWalk([0.2, 1.0], [false, true])]
    ρ = 0.5
    param_updt = true
    updt_coord = (Val((true,true,false)),
                  Val((false,true,true)))
    updt_type=(MetropolisHastingsUpdt(),
               ConjugateUpdt())
    set_transition_kernels!(setup, t_kernels, ρ, param_updt, updt_coord,
                            updt_type)

    priors = Priors((ImproperPrior(), ImproperPrior()))
    x0_prior = KnownStartingPt(obs[1])
    set_priors!(setup, priors, x0_prior)

    num_mcmc_steps = 100
    set_mcmc_params!(setup, num_mcmc_steps)

    set_solver!(setup)

    initialise!(Float64, setup)

    ws = Workspace(setup).workspace
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
        @test typeof(ws.P[1].Target) <: FitzhughDiffusion
        @test typeof(ws.P[1].Pt) <: FitzhughDiffusionAux
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
        @test !act(SavePath(), ws, 1)
        @test !act(SavePath(), ws, 10)
        @test !act(SavePath(), ws, 60)
        @test !act(SavePath(), ws, 51)
        @test !act(SavePath(), ws, 61)
        @test !act(SavePath(), ws, 40000)

        @test !act(Verbose(), ws, 1)
        @test !act(Verbose(), ws, 3)
        @test !act(Verbose(), ws, 10)
        @test !act(Verbose(), ws, 30)

        @test act(ParamUpdate(), ws, 1)
        @test act(ParamUpdate(), ws, 10)
        @test act(ParamUpdate(), ws, 50)
        @test act(ParamUpdate(), ws, 51)
        @test act(ParamUpdate(), ws, 1000)
    end
end



@testset "definition of gibbs sweep" begin
    out = init_setup()
    setup, obs, tt = out.setup, out.obs, out.tt
    L = [1. 0.; 0. 1.]
    Σ = [0.5 0.0; 0.0 1.0]
    set_observations!(setup, [L, L], [Σ, 2*Σ], obs, tt)

    dt = 0.01
    τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
    set_imputation_grid!(setup, dt, τ)

    t_kernels = [RandomWalk([0.002, 0.1], [true, true]),
                 RandomWalk([0.2, 1.0], [false, true])]
    ρ = 0.5
    param_updt = true
    updt_coord = (Val((true,true,false)),
                  Val((false,true,true)))
    updt_type=(MetropolisHastingsUpdt(),
               ConjugateUpdt())
    set_transition_kernels!(setup, t_kernels, ρ, param_updt, updt_coord,
                            updt_type)

    priors = Priors((ImproperPrior(), ImproperPrior()))
    x0_prior = KnownStartingPt(obs[1])
    set_priors!(setup, priors, x0_prior)

    num_mcmc_steps = 100
    set_mcmc_params!(setup, num_mcmc_steps)

    set_solver!(setup)

    initialise!(Float64, setup)
    gs = GibbsDefn(setup)
    @testset "initialisation" begin
        @test length(gs) == 2
        @test all([gs[i] == gs.updates[i] for i in 1:2])
        @test all([gs[i].updt_type == updt_type[i] for i in 1:2])
        @test all([gs[i].updt_coord == updt_coord[i] for i in 1:2])
        @test all([gs[i].priors == priors[i] for i in 1:2])
        @test gs[1].recompute_ODEs && gs[2].recompute_ODEs
    end
end
