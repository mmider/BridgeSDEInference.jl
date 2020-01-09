SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)
#NOTE additional packages that need installing:
#=
    CSV, DataFrames, RollingFunctions

=#



#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
include(joinpath(SRC_DIR, "BridgeSDEInference_for_tests.jl"))
#using BridgeSDEInference
#const BSI = BridgeSDEInference
using Distributions # to define priors
using Random        # to seed the random number generator
using DataFrames
using CSV
using LinearAlgebra

DIR = "auxiliary"
include(joinpath(SRC_DIR, DIR, "read_and_write_data.jl"))
include(joinpath(SRC_DIR, DIR, "transforms.jl"))
include(joinpath(SRC_DIR, DIR, "utility_functions.jl"))

# decide if first passage time observations or partially observed diffusion
fptObsFlag = false

# simulate dataset
include(joinpath(SRC_DIR, "..", "scripts", "data_generation",
                 "simulate_repeated_fhn.jl"))

param = :complexConjug
θ_init = [10.0, -8.0, 15.0, 0.0, 3.0]
P˟ = FitzhughDiffusion(param, θ_init...)
P̃s = [std_aux_laws(FitzhughDiffusionAux, param, θ_init, map(λ->λ[1], xx.yy), xx.tt, 1) for xx in XX]
L = @SMatrix [1. 0.]
Σ = @SMatrix [10^(-10)]

model_setups = [DiffusionSetup(P˟, P̃, PartObs()) for P̃ in P̃s]
for (k,(model_setup, P̃, xx))  in enumerate(zip(model_setups, P̃s, XX))
      obs, obs_times = map(λ->λ[1], xx.yy), xx.tt
      set_observations!(model_setup, [L for _ in P̃], [Σ for _ in P̃], obs, obs_times)
      set_imputation_grid!(model_setup, 1/1000)
      set_x0_prior!(model_setup, KnownStartingPt(x0[k]))
      initialise!(eltype(x0[k]), model_setup, Vern7(), false, NoChangePt(100))
      set_auxiliary!(model_setup; skip_for_save=10^0, adaptive_prop=NoAdaptation())
end

mcmc_setup = MCMCSetup(
      Imputation(NoBlocking(), 0.975, Vern7()),
      ParamUpdate(MetropolisHastingsUpdt(), 5, θ_init,
                  UniformRandomWalk(0.5, true), ImproperPosPrior(),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃s[1], 5))
                  ),
      ParamUpdate(ConjugateUpdt(), [1,2,3], θ_init, nothing,
                  MvNormal(fill(0.0, 3), diagm(0=>fill(1000.0, 3))),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃s[1], [1,2,3]))
                  ))

schedule = MCMCSchedule(1*10^4, [[1,2,3]],
                        (save=3*10^2, verbose=10^2, warm_up=100,
                         readjust=(x->x%100==0), fuse=(x->false)))

Random.seed!(4)
out, elapsed = @timeit mcmc(mcmc_setup, schedule, model_setups)
#out = mcmc(mcmc_setup, schedule, model_setup)

include(joinpath(SRC_DIR, DIR, "plotting_fns.jl"))
plot_chains(out[2]; truth=[10.0, -8.0, 15.0, 0.0, 3.0])
plot_paths(out[1], out[2], schedule; transf=[(x,θ)->x, (x,θ)->conjugToRegular(x, θ[1], 0)],
           obs=(times=obs_time[2:end], vals=obs[2:end], indices=1))
