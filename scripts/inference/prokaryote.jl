SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
include(joinpath(SRC_DIR, "BridgeSDEInference_for_tests.jl"))


using StaticArrays
using Distributions # to define priors
using Random        # to seed the random number generator

# Let's generate the data
# -----------------------
using Bridge
include(joinpath(SRC_DIR, "auxiliary", "data_simulation_fns.jl"))
Random.seed!(4)
# values taken from table 1 of Golithly and Wilkinson
θ_init = [0.1, 0.7, 0.35, 0.2, 0.1, 0.9, 0.3, 0.1]
K = 10.0
Pˣ = Prokaryote(θ_init..., K)

x0, dt, T = ℝ{4}(7.0, 10.0, 4.0, 6.0), 1/5000, 20.0
tt = 0.0:dt:T
XX, _ = simulateSegment(ℝ{4}(0.0, 0.0, 0.0, 0.0), x0, Pˣ, tt)
skip = 2500
obs_time, obs_vals = XX.tt[1:skip:end], XX.yy[1:skip:end]

auxFlag = Val{(true, true, true, true)}()
P̃ = [ProkaryoteAux(θ_init..., K, t₀, u, T, v, auxFlag) for (t₀, T, u, v)
     in zip(obs_time[1:end-1], obs_time[2:end], obs[1:end-1], obs[2:end])]

Σdiagel = 10^-4
Σ = SMatrix{4,4}(1.0I)*Σdiagel
L = SMatrix{4,4}(1.0I)

setup = MCMCSetup(P˟, P̃, PartObs())
set_observations!(setup, [L for _ in P̃], [Σ for _ in P̃], obs, obs_time)
set_imputation_grid!(setup, 1/1000)
set_transition_kernels!(setup,
                        [RandomWalk([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                                    collect(1:8))],
                        0.7, false,
                        (Val((true, false, false, false, false, false, false, false)),),
                        (MetropolisHastingsUpdt(),))
set_priors!(setup,
            Priors((ImproperPrior(), ImproperPrior(), ImproperPrior(),
                    ImproperPrior(), ImproperPrior(), ImproperPrior(),
                    ImproperPrior(), ImproperPrior())),
            KnownStartingPt(x0)
            )
set_mcmc_params!(setup, 1*10^3, 1*10^0, 10^1, 10^0, 0)
set_blocking!(setup)
set_solver!(setup, Vern7(), NoChangePt())
initialise!(eltype(x0), setup)

Random.seed!(4)
out, elapsed = @timeit mcmc(setup)
display(out.accpt_tracker)

pTp = [[[x[i] for x in path] for path in paths] for i in 1:4]

using Plots

function plotPaths(j, obsIdxS, obsIdxE)
    idxS = div((obsIdxS-1)*skip,5)+1
    idxE = div((obsIdxE-1)*skip,5)+1
    p = plot()
    for i in 1:length(paths)
        plot!(time_[idxS:idxE], pTp[j][i][idxS:idxE], label="", color="steelblue", alpha=0.3)
    end
    scatter!(obsTime[obsIdxS:obsIdxE], [x[j] for x in obsVals][obsIdxS:obsIdxE],
             color="orange", label="")
    p
end
plotPaths(1, 1, 5)
plotPaths(2, 1, 5)
plotPaths(3, 1, 5)
plotPaths(4, 1, 5)

plot([θ[1] for θ in chain])
