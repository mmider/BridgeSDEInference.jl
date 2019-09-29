SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference

include(joinpath(SRC_DIR, "types.jl"))

DIR = "stochastic_process"
include(joinpath(SRC_DIR, DIR, "bounded_diffusion_domain.jl"))
include(joinpath(SRC_DIR, DIR, "guid_prop_bridge.jl"))
include(joinpath(SRC_DIR, DIR, "path_to_wiener.jl"))

DIR = "solvers"
include(joinpath(SRC_DIR, DIR, "vern7.jl"))
include(joinpath(SRC_DIR, DIR, "tsit5.jl"))
include(joinpath(SRC_DIR, DIR, "rk4.jl"))
include(joinpath(SRC_DIR, DIR, "ralston3.jl"))
include(joinpath(SRC_DIR, DIR, "euler_maruyama_dom_restr.jl"))

DIR = "transition_kernels"
include(joinpath(SRC_DIR, DIR, "random_walk.jl"))

DIR = "mcmc_extras"
include(joinpath(SRC_DIR, DIR, "adaptation.jl"))
include(joinpath(SRC_DIR, DIR, "blocking_schedule.jl"))
include(joinpath(SRC_DIR, DIR, "first_passage_times.jl"))
include(joinpath(SRC_DIR, DIR, "starting_pt.jl"))

DIR = "mcmc"
include(joinpath(SRC_DIR, DIR, "priors.jl"))
include(joinpath(SRC_DIR, DIR, "setup.jl"))
include(joinpath(SRC_DIR, DIR, "workspace.jl"))
include(joinpath(SRC_DIR, DIR, "conjugateUpdt.jl"))
include(joinpath(SRC_DIR, DIR, "mcmc.jl"))

DIR = "examples"
include(joinpath(SRC_DIR, DIR, "fitzHughNagumo.jl"))


using Distributions # to define priors
using Random        # to seed the random number generator
using DataFrames
using CSV

DIR = "auxiliary"
include(joinpath(SRC_DIR, DIR, "read_and_write_data.jl"))
include(joinpath(SRC_DIR, DIR, "transforms.jl"))

# decide if first passage time observations or partially observed diffusion
fptObsFlag = false

# pick dataset
filename = "path_part_obs_conj.csv"

# fetch the data
(df, x0, obs, obs_time, fpt,
      fptOrPartObs) = readData(Val(fptObsFlag), joinpath(OUT_DIR, filename))

param = :complexConjug
θ₀ = [10.0, -8.0, 15.0, 0.0, 3.0]
P˟ = FitzhughDiffusion(param, θ₀...)
P̃ = [FitzhughDiffusionAux(param, θ₀..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
     in zip(obs_time[1:end-1], obs_time[2:end], obs[1:end-1], obs[2:end])]
display(P̃[1])
setup = MCMCSetup(P˟, P̃, fptOrPartObs)

L = @SMatrix [1. 0.]
Σ = @SMatrix [10^(-10)]
set_observations!(setup, [L for _ in P̃], [Σ for _ in P̃], obs, obs_time, fpt)

set_imputation_grid!(setup, 1/1000)

ρ = 0.975
set_transition_kernels!(setup,
                        [RandomWalk([],[]),
                         RandomWalk([0.0, 0.0, 0.0, 0.0, 0.5],
                                    [false, false, false, false, true])],
                        ρ, true,
                        (Val((true, true, true, false, false)),
                         Val((false, false, false, false, true)),
                         ),
                        (ConjugateUpdt(),
                         MetropolisHastingsUpdt(),
                         ))

set_priors!(setup,
            Priors((MvNormal([0.0,0.0,0.0],
                             diagm(0=>[1000.0, 1000.0, 1000.0])),
                    ImproperPrior(),
                    )),
            GsnStartingPt(x0, x0, @SMatrix [20. 0; 0 20.])
            )
# num_mcmc_steps, save_iter, verb_iter, skip_for_save, warm_up
set_mcmc_params!(setup, 1*10^4, 3*10^2, 10^2, 10^0, 100)
set_blocking!(setup)
set_solver!(setup, Vern7(), NoChangePt())
initialise!(eltype(x0), setup)

Random.seed!(4)
start = time()
(chain, accRateImp, accRateUpdt, paths, time_) = mcmc(setup)
elapsed = time() - start
print("time elapsed: ", elapsed, "\n")

print("imputation acceptance rate: ", accRateImp,
      ", parameter update acceptance rate: ", accRateUpdt)

x0⁺, pathsToSave = transformMCMCOutput(x0, paths, saveIter; chain=chain,
                                       numGibbsSteps=2,
                                       parametrisation=param,
                                       warmUp=warmUp)


df2 = savePathsToFile(pathsToSave, time_, joinpath(OUT_DIR, "sampled_paths.csv"))
df3 = saveChainToFile(chain, joinpath(OUT_DIR, "chain.csv"))


include(joinpath(AUX_DIR, "plotting_fns.jl"))
set_default_plot_size(30cm, 20cm)
plotPaths(df2, obs=[Float64.(df.x1), [x0⁺[2]]],
          obsTime=[Float64.(df.time), [0.0]], obsCoords=[1,2])

plotChain(df3, coords=[1])
plotChain(df3, coords=[2])
plotChain(df3, coords=[3])
plotChain(df3, coords=[5])
