using Suppressor

include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "mcmc", "setup.jl"))
include(joinpath(SRC_DIR, "examples", "fitzHughNagumo.jl"))
include(joinpath(SRC_DIR, "transition_kernels", "random_walk.jl"))
include(joinpath(SRC_DIR, "mcmc_extras", "adaptation.jl"))
include(joinpath(SRC_DIR, "mcmc", "priors.jl"))
include(joinpath(SRC_DIR, "stochastic_process", "guid_prop_bridge.jl"))
include(joinpath(SRC_DIR, "mcmc_extras", "starting_pt.jl"))
include(joinpath(SRC_DIR, "solvers", "ralston3.jl"))

@testset "setup object" begin
    param = :complexConjug
    θ₀ = [10.0, -5.0, 5.0, 0.0, 3.0]
    P˟ = FitzhughDiffusion(param, θ₀...)
    obs = [[0.0, 0.0], [-0.3, -0.2], [0.4, 0.2]]
    tt = [0.0, 1.0, 1.5]
    P̃ = [FitzhughDiffusionAux(param, θ₀..., tt[1], obs[1], tt[2], obs[2]),
         FitzhughDiffusionAux(param, θ₀..., tt[2], obs[2], tt[3], obs[3])]
    setup = MCMCSetup(P˟, P̃, PartObs())

    @testset "initialisation" begin
        @test setup.P˟ == P˟
        @test setup.P̃ == P̃
        @test setup.blocking == NoBlocking()
        @test setup.blocking_params == ([], 0.1, NoChangePt())
        @test !any(values(setup.setup_completion))
        @test @suppress !check_if_complete(setup, [:obs])
        @test @suppress !check_if_complete(setup, [:imput])
        @test @suppress !check_if_complete(setup, [:tkern])
        @test @suppress !check_if_complete(setup, [:prior])
        @test @suppress !check_if_complete(setup, [:mcmc])
        @test @suppress !check_if_complete(setup, [:solv])
    end

    L = [1. 0.; 0. 1.]
    Σ = [0.5 0.0; 0.0 1.0]
    set_observations!(setup, [L, L], [Σ, 2*Σ], obs, tt)

    @testset "setting observations" begin
        @test setup.Ls == [L, L]
        @test setup.Σs == [Σ, 2*Σ]
        @test setup.obs == obs
        @test setup.obs_times == tt
        @test setup.fpt == [nothing, nothing]
        @test @suppress !check_if_complete(setup, [:imput])
        @test @suppress !check_if_complete(setup, [:tkern])
        @test @suppress !check_if_complete(setup, [:prior])
        @test @suppress !check_if_complete(setup, [:mcmc])
        @test @suppress !check_if_complete(setup, [:solv])
        @test @suppress check_if_complete(setup, [:obs])
    end

    dt = 0.01
    τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
    set_imputation_grid!(setup, dt)

    @testset "setting imputation grid" begin
        @test setup.dt == dt
        @test setup.τ(tt[1], tt[2])(0.5*(tt[1]+tt[2])) == τ(tt[1], tt[2])(0.5*(tt[1]+tt[2]))
        @test @suppress !check_if_complete(setup, [:tkern])
        @test @suppress !check_if_complete(setup, [:prior])
        @test @suppress !check_if_complete(setup, [:mcmc])
        @test @suppress !check_if_complete(setup, [:solv])
        @test @suppress check_if_complete(setup, [:obs, :imput])
    end

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
    @testset "setting transition kernels" begin
        @test setup.t_kernel == t_kernels
        @test setup.ρ == ρ
        @test setup.param_updt == param_updt
        @test setup.updt_coord == updt_coord
        @test setup.updt_type == updt_type
        @test !check_if_adapt(setup.adaptive_prop)
        @test @suppress !check_if_complete(setup, [:prior])
        @test @suppress !check_if_complete(setup, [:mcmc])
        @test @suppress !check_if_complete(setup, [:solv])
        @test @suppress check_if_complete(setup, [:obs, :imput, :tkern])
    end

    priors = Priors((ImproperPrior(), ImproperPrior()))
    x0_prior = KnownStartingPt(obs[1])
    set_priors!(setup, priors, x0_prior)
    @testset "setting priors" begin
        @test setup.priors == priors
        @test setup.x0_prior == x0_prior
        @test @suppress !check_if_complete(setup, [:mcmc])
        @test @suppress !check_if_complete(setup, [:solv])
        @test @suppress check_if_complete(setup, [:obs, :imput, :tkern, :prior])
    end

    num_mcmc_steps = 100
    set_mcmc_params!(setup, num_mcmc_steps)
    @testset "setting  mcmc parameters" begin
        @test setup.num_mcmc_steps == num_mcmc_steps
        @test isnan(setup.save_iter)
        @test isnan(setup.verb_iter)
        @test setup.skip_for_save == 1
        @test setup.warm_up == 0
        @test @suppress !check_if_complete(setup, [:solv])
        @test @suppress check_if_complete(setup, [:obs, :imput, :tkern, :prior,
                                                  :mcmc])
    end

    set_solver!(setup)
    @testset "setting solver" begin
        @test setup.solver == Ralston3()
        @test setup.change_pt == NoChangePt()
        @test @suppress check_if_complete(setup)
    end

    @testset "determining data type" begin
        @test determine_data_type(setup) == (SArray{Tuple{2},Float64,1,2},
                                             Float64)
    end

    prepare_containers!(setup)
    @testset "setting internal containers" begin
        @test setup.Wnr == Wiener{Float64}()
        @test eltype(setup.WW) == SamplePath{Float64}
        @test length(setup.WW) == 2
        @test eltype(setup.XX) == SamplePath{SArray{Tuple{2},Float64,1,2}}
        @test length(setup.XX) == 2
        @test setup.Ls == [SMatrix{2,2}(L), SMatrix{2,2}(L)]
        @test setup.Σs == [SMatrix{2,2}(Σ), SMatrix{2,2}(2*Σ)]
        @test setup.obs == map(x->SVector{2}(x), obs)
    end

    initialise!(Float64, setup)
    @testset "initialisation of proposal law" begin
        @test length(setup.P) == 2
        @test typeof(setup.P[1].Pt) == typeof(setup.P̃[1])
        @test setup.P[1].Target == setup.P˟
    end
end
