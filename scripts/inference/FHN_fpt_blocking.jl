SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
include(joinpath(SRC_DIR, "BridgeSDEInference_for_tests.jl"))

using Distributions # to define priors
using Random        # to seed the random number generator
using DataFrames
using CSV

DIR = "auxiliary"
include(joinpath(SRC_DIR, DIR, "read_and_write_data.jl"))
include(joinpath(SRC_DIR, DIR, "transforms.jl"))
include(joinpath(SRC_DIR, DIR, "utility_functions.jl"))

# decide if first passage time observations or partially observed diffusion
fptObsFlag = true

# pick dataset
filename = "path_fpt_simpleConjug.csv"

# fetch the data
(df, x0, obs, obs_time, fpt,
      fptOrPartObs) = readData(Val(fptObsFlag), joinpath(OUT_DIR, filename))
x0
param = :complexConjug
θ_init = [10.0, -8.0, 25.0, 0.0, 3.0]
P˟ = FitzhughDiffusion(param, θ_init...)
P̃ = std_aux_laws(FitzhughDiffusionAux, param, θ_init, obs, obs_time, 1)
L = @SMatrix [1. 0.]
Σ = @SMatrix [10^(-10)]

setup = MCMCSetup(P˟, P̃, fptOrPartObs)
set_observations!(setup, [L for _ in P̃], [Σ for _ in P̃], obs, obs_time, fpt)
set_imputation_grid!(setup, 1/5000)
set_transition_kernels!(setup,
                        [RandomWalk([3.0, 5.0, 1.0, 0.01, 0.5], 5)],
                        0.99, true, 3,
                        (MetropolisHastingsUpdt(),))
set_priors!(setup, Priors((ImproperPrior(),)), KnownStartingPt(x0))
set_mcmc_params!(setup, 3*10^4, 3*10^2, 10^2, 10^1, 100)
set_blocking!(setup, ChequeredBlocking(),
              (collect(1:length(obs)-2)[1:1:end], 10^(-10), SimpleChangePt(100)))
set_solver!(setup, Vern7(), NoChangePt())
initialise!(eltype(x0), setup)

Random.seed!(4)
out, elapsed = @timeit mcmc(setup)
display(out.accpt_tracker)

include(joinpath(SRC_DIR, DIR, "plotting_fns.jl"))
plot_chains(out; truth=[10.0, -8.0, 25.0, 0.0, 3.0])
plot_paths(out; transf=[(x,θ)->x, (x,θ)->conjugToRegular(x, θ[1], 0)],
           obs=(times=obs_time[2:end], vals=obs[2:end], indices=1))
