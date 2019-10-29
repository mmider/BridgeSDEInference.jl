SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
include(joinpath(SRC_DIR, "BridgeSDEInference_for_tests.jl"))


using StaticArrays
using Distributions
using Random
ImproperPrior()
DIR = "auxiliary"
include(joinpath(SRC_DIR, DIR, "utility_functions.jl"))

# Let's load-in the data
# -----------------------
obs = open(joinpath(OUT_DIR, "autoreg50fo.dat")) do f
    [map(x->parse(Float64, x), split(l, ' '))[[2,3,4,1]] for l in eachline(f)]
end
obs_time = collect(range(0.0, length(obs)-1, step=1))

# let's start from the true values that generated the data
θ_init = [0.1, 0.7, 0.35, 0.2, 0.1, 0.9, 0.3, 0.1]
K = 10.0
P˟ = Prokaryote(θ_init..., K)
x0 = ℝ{4}(obs[1])

auxFlag = Val{(true, true, true, true)}()
P̃ = [ProkaryoteAux(θ_init..., K, t₀, u, T, v, auxFlag) for (t₀, T, u, v)
     in zip(obs_time[1:end-1], obs_time[2:end], obs[1:end-1], obs[2:end])]

Σdiagel = 10^-1
Σ = SMatrix{4,4}(1.0I)*Σdiagel
L = SMatrix{4,4}(1.0I)

setup = MCMCSetup(P˟, P̃, PartObs())
set_observations!(setup, [L for _ in P̃], [Σ for _ in P̃], obs, obs_time)
set_imputation_grid!(setup, 1/400)
set_transition_kernels!(setup,
                        [RandomWalk([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                    collect(1:8)) for _ in 1:8],
                        0.9, true, [[1], [2], [3], [4], [5], [6], [7], [8]],
                        (MetropolisHastingsUpdt(),
                        MetropolisHastingsUpdt(),
                        MetropolisHastingsUpdt(),
                        MetropolisHastingsUpdt(),
                        MetropolisHastingsUpdt(),
                        MetropolisHastingsUpdt(),
                        MetropolisHastingsUpdt(),
                        MetropolisHastingsUpdt()))
set_priors!(setup,
            Priors((ImproperPosPrior{Val{1}}(), ImproperPosPrior{Val{2}}(),
                    ImproperPosPrior{Val{3}}(), ImproperPosPrior{Val{4}}(),
                    ImproperPosPrior{Val{5}}(), ImproperPosPrior{Val{6}}(),
                    ImproperPosPrior{Val{7}}(), ImproperPosPrior{Val{8}}())),
            KnownStartingPt(x0)
            )
set_mcmc_params!(setup, 1*10^3, 1*10^2, 10^1, 10^0, 0,
                 (50, 0.1, 0.00001, 0.99999, 0.234, 50),
                 (50, 0.1, -999, 999, 0.234, 50, (1,2,3,4,5,6,7,8), (1,2,3,4,5,6,7,8)))
set_blocking!(setup)
set_solver!(setup, Vern7(), NoChangePt())
initialise!(eltype(x0), setup)

Random.seed!(4)
out, elapsed = @timeit mcmc(setup)
display(out.accpt_tracker)

include(joinpath(SRC_DIR, DIR, "plotting_fns.jl"))
plot_paths(out; obs=(times=obs_time[2:end], vals=obs[2:end], indices=1:4))
plot_chains(out; truth=[0.1, 0.7, 0.35, 0.2, 0.1, 0.9, 0.3, 0.1])
