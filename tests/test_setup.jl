
include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "mcmc", "setup.jl"))
include(joinpath(SRC_DIR, "examples", "fitzHughNagumo.jl"))

@testset "setup object" begin
    param = :complexConjug
    θ₀ = [10.0, -5.0, 5.0, 0.0, 3.0]
    P˟ = FitzhughDiffusion(param, θ₀...)
    P̃ = [FitzhughDiffusionAux(param, θ₀..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
         in zip([0.0, 1.0], obsTime[2:end], obs[1:end-1], obs[2:end])]
end




include(joinpath(SRC_DIR, "mcmc_extras", "blocking_schedule.jl"))




include(joinpath(SRC_DIR, "stochastic_process", "guid_prop_bridge.jl"))

include(joinpath(SRC_DIR, "mcmc_extras", "starting_pt.jl"))
include(joinpath(SRC_DIR, "mcmc_extras", "adaptation.jl"))
