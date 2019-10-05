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

x0⁺, pathsToSave = transformMCMCOutput(x0, paths, saveIter; chain=chain, #chain=chain #θ=θ₀
                                       numGibbsSteps=1,
                                       parametrisation=param,
                                       warmUp=warmUp)

df2 = savePathsToFile(pathsToSave, time_, joinpath(OUT_DIR, "sampled_paths.csv"))
df3 = saveChainToFile(chain, joinpath(OUT_DIR, "chain.csv"))

include(joinpath(AUX_DIR, "plotting_fns.jl"))
set_default_plot_size(30cm, 20cm)
plotPaths(df2, obs=[Float64.(df.upCross), [x0⁺[2]]],
          obsTime=[Float64.(df.time), [0.0]], obsCoords=[1,2])

plotChain(df3, coords=[1])
plotChain(df3, coords=[2])
plotChain(df3, coords=[3])
plotChain(df3, coords=[5])
