SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
include(joinpath(SRC_DIR, "BridgeSDEInference_for_tests.jl"))

using StaticArrays
using Distributions
using Random
using DataFrames
using CSV

DIR = "auxiliary"
include(joinpath(SRC_DIR, DIR, "read_and_write_data.jl"))
include(joinpath(SRC_DIR, DIR, "transforms.jl"))
include(joinpath(SRC_DIR, DIR, "utility_functions.jl"))

# decide if first passage time observations or partially observed diffusion
fptObsFlag = false

# pick dataset
filename = "path_part_obs_conj.csv"

# fetch the data
(df, x0, obs, obs_time, fpt,
      fpt_or_part_obs) = readData(Val(fptObsFlag), joinpath(OUT_DIR, filename))

param = :complexConjug
θ_init = [10.0, -8.0, 15.0, 0.0, 3.0]
P˟ = FitzhughDiffusion(param, θ_init...)
P̃ = std_aux_laws(FitzhughDiffusionAux, param, θ_init, obs, obs_time, 1)
L = @SMatrix [1. 0.]
Σ = @SMatrix [10^(-10)]

model_setup = DiffusionSetup(P˟, P̃, fptOrPartObs)
set_observations!(model_setup, [L for _ in P̃], [Σ for _ in P̃], obs, obs_time, fpt)
set_imputation_grid!(model_setup, 1/1000)
set_x0_prior!(model_setup, GsnStartingPt(x0, @SMatrix [1. 0; 0 1.]), x0)
initialise!(eltype(x0), model_setup, Vern7(), false, NoChangePt(100))
set_auxiliary!(model_setup; skip_for_save=10^0, adaptive_prop=NoAdaptation())

blocks = create_blocks( ChequeredBlocking(), model_setup.P,
                        (knots=collect(1:length(obs)-2)[1:1:end],
                         ϵ=10^(-7),
                         change_pt=SimpleChangePt(100)) )

mcmc_setup = MCMCSetup(
      Imputation(blocks[1], 0.6, Vern7()),
      Imputation(blocks[2], 0.6, Vern7()))#,
#      ParamUpdate(MetropolisHastingsUpdt(), 5, θ_init,
#                  UniformRandomWalk(0.5, true), ImproperPosPrior(),
#                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, 5))
#                  ),
#      ParamUpdate(ConjugateUpdt(), [1,2,3], θ_init, nothing,
#                  MvNormal(fill(0.0, 3), diagm(0=>fill(1000.0, 3))),
#                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, [1,2,3]))
#                  ))

schedule = MCMCSchedule(1*10^4, [[1,2]],
                        (save=3*10^2, verbose=10^2, warm_up=100,
                         readjust=(x->x%100==0), fuse=(x->false)))


Random.seed!(4)
out, elapsed = @timeit mcmc(mcmc_setup, schedule, model_setup)
#[i for (i,v) in enumerate(diff([v[2] for v in out[1].paths[10]])) if v>0.6]
#[v[2] for v in out[1].paths[1]][195:210]
#maximum(diff([v[2] for v in out[1].paths[]]))

include(joinpath(SRC_DIR, DIR, "plotting_fns.jl"))
plot_chains(out[2]; truth=[10.0, -8.0, 15.0, 0.0, 3.0])
plot_paths(out[1], out[2], schedule; transf=[(x,θ)->x, (x,θ)->conjugToRegular(x, θ[1], 0)],
           obs=(times=obs_time[2:end], vals=obs[2:end], indices=1))
