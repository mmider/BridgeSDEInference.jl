SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
include(joinpath(SRC_DIR, "BridgeSDEInference_for_tests.jl"))

using Distributions, Random
using DataFrames, CSV

DIR = "auxiliary"
include(joinpath(SRC_DIR, DIR, "read_and_write_data.jl"))
include(joinpath(SRC_DIR, DIR, "transforms.jl"))
include(joinpath(SRC_DIR, DIR, "utility_functions.jl"))

using LinearAlgebra

# decide if first passage time observations or partially observed diffusion
fptObsFlag = true

# pick dataset
filename = "test_path_fpt_simpleConjug.csv"#"path_fpt_simpleConjug.csv"

# fetch the data
(df, x0, obs, obs_time, fpt,
      fpt_or_part_obs) = readData(Val(fptObsFlag), joinpath(OUT_DIR, filename))

param = :complexConjug
θ_init = [10.0, -5.0, 5.0, 0.0, 3.0]
P˟ = FitzhughDiffusion(param, θ₀...)
P̃ = std_aux_laws(FitzhughDiffusionAux, param, θ_init, obs, obs_time, 1)
L = @SMatrix [1. 0.]
Σ = @SMatrix [10^(-10)]

setup = MCMCSetup(P˟, P̃, fpt_or_part_obs)
set_observations!(setup, [L for _ in P̃], [Σ for _ in P̃], obs, obs_time, fpt)
set_imputation_grid!(setup, 1/2000)
set_transition_kernels!(setup,
                        [RandomWalk([3.0, 5.0, 0.5, 0.01, 0.5], 5)],
                        0.9995, true, (2,3),
                        (ConjugateUpdt(),))
set_priors!(setup,
            Priors((MvNormal([0.0,0.0], diagm(0=>[1000.0, 1000.0])),
                    )),
            GsnStartingPt(zero(typeof(x0)), zero(typeof(x0)), @SMatrix [100. 0; 0 100.])
            )
set_mcmc_params!(setup, 3*10^5, 1*10^4, 10^2, 10^1, 100)
set_blocking!(setup)
set_solver!(setup, Vern7(), NoChangePt())
initialise!(eltype(x0), setup)

Random.seed!(4)
out, elapsed = @timeit mcmc(setup)
display(out.accpt_tracker)

include(joinpath(SRC_DIR, DIR, "plotting_fns.jl"))
plot_chains(out; truth=[10.0, -8.0, 15.0, 0.0, 3.0]) # this might not be the truth
plot_paths(out; transf=[(x,θ)->x, (x,θ)->conjugToRegular(x, θ[1], 0)],
           obs=(times=obs_time[2:end], vals=obs[2:end], indices=1))
