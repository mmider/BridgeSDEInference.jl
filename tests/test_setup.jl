using Suppressor

include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "mcmc", "setup.jl"))
include(joinpath(SRC_DIR, "examples", "fitzHughNagumo.jl"))
include(joinpath(SRC_DIR, "transition_kernels", "random_walk.jl"))
include(joinpath(SRC_DIR, "mcmc_extras", "adaptation.jl"))

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
        @test @suppress !check_if_complete(setup, [:obs, :imput, :tkern, :prior,
                                                   :mcmc, :solv])
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
        @test @suppress !check_if_complete(setup, [:imput, :tkern, :prior,
                                                         :mcmc, :solv])
        @test @suppress check_if_complete(setup, [:obs])
    end

    dt = 0.01
    τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
    set_imputation_grid!(setup, dt, τ)

    @testset "setting imputation grid" begin
        @test setup.dt == dt
        @test setup.τ == τ
        @test @suppress !check_if_complete(setup, [:tkern, :prior, :mcmc, :solv]
                                          )
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
        @test @suppress !check_if_complete(setup, [:prior, :mcmc, :solv])
        @test @suppress check_if_complete(setup, [:obs, :imput, :tkern])
    end

    

end


#include(joinpath(SRC_DIR, "mcmc_extras", "blocking_schedule.jl"))

#include(joinpath(SRC_DIR, "mcmc_extras", "blocking_schedule.jl"))




#include(joinpath(SRC_DIR, "stochastic_process", "guid_prop_bridge.jl"))

#include(joinpath(SRC_DIR, "mcmc_extras", "starting_pt.jl"))
#include(joinpath(SRC_DIR, "mcmc_extras", "adaptation.jl"))
