
include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "mcmc", "setup.jl"))
include(joinpath(SRC_DIR, "examples", "fitzHughNagumo.jl"))


@testset "setup object" begin
    param = :complexConjug
    θ₀ = [10.0, -5.0, 5.0, 0.0, 3.0]
    P˟ = FitzhughDiffusion(param, θ₀...)
    obs = [[0.0, 0.0], [-0.3, -0.2], [0.4, 0.2]]
    P̃ = [FitzhughDiffusionAux(param, θ₀..., 0.0, obs[1], 1.0, obs[2]),
         FitzhughDiffusionAux(param, θ₀..., 1.0, obs[2], 1.5, obs[3])]
    setup = MCMCSetup(P˟, P̃, PartObs())

    @testset "initialisation" begin
        @test setup.P˟ == P˟
        @test setup.P̃ == P̃
        @test setup.blocking == NoBlocking()
        @test setup.blocking_params == ([], 0.1, NoChangePt())
        @test !any(values(setup.setup_completion))
    end
end


#include(joinpath(SRC_DIR, "mcmc_extras", "blocking_schedule.jl"))

#include(joinpath(SRC_DIR, "mcmc_extras", "blocking_schedule.jl"))




#include(joinpath(SRC_DIR, "stochastic_process", "guid_prop_bridge.jl"))

#include(joinpath(SRC_DIR, "mcmc_extras", "starting_pt.jl"))
#include(joinpath(SRC_DIR, "mcmc_extras", "adaptation.jl"))
