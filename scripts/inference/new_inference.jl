SRC_DIR = joinpath(Base.source_dir(), "..", "src")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
#include(joinpath(SRC_DIR, "fitzHughNagumo.jl"))
#include(joinpath(SRC_DIR, "fitzHughNagumo_conjugateUpdt.jl"))

include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "solvers", "vern7.jl"))
include(joinpath(SRC_DIR, "solvers", "tsit5.jl"))
include(joinpath(SRC_DIR, "solvers", "rk4.jl"))
include(joinpath(SRC_DIR, "solvers", "ralston3.jl"))
include(joinpath(SRC_DIR, "mcmc", "priors.jl"))

include(joinpath(SRC_DIR, "stochastic_process", "bounded_diffusion_domain.jl"))
include(joinpath(SRC_DIR, "examples", "lorenz_system.jl"))
include(joinpath(SRC_DIR, "examples", "lorenz_system_const_vola.jl"))
include(joinpath(SRC_DIR, "solvers", "euler_maruyama_dom_restr.jl"))

include(joinpath(SRC_DIR, "stochastic_process", "guid_prop_bridge.jl"))
include(joinpath(SRC_DIR, "transition_kernels", "random_walk.jl"))
include(joinpath(SRC_DIR, "mcmc_extras", "first_passage_times.jl"))
include(joinpath(SRC_DIR, "mcmc_extras", "blocking_schedule.jl"))
include(joinpath(SRC_DIR, "mcmc_extras", "starting_pt.jl"))
include(joinpath(SRC_DIR, "mcmc_extras", "adaptation.jl"))
include(joinpath(SRC_DIR, "mcmc", "setup.jl"))
include(joinpath(SRC_DIR, "mcmc", "mcmc.jl"))
include(joinpath(SRC_DIR, "stochastic_process", "path_to_wiener.jl"))


using StaticArrays
using Distributions # to define priors
using Random        # to seed the random number generator
# Let's generate the data
# -----------------------
using Bridge
include(joinpath(SRC_DIR, "auxiliary", "data_simulation_fns.jl"))
Random.seed!(4)
#θ₀ = [10.0, 28.0, 8.0/3.0, 3.0, 3.0, 3.0]
θˣ = [10.0, 28.0, 8.0/3.0, 3.0]
Pˣ = LorenzCV(θˣ...)

x0, dt, T = ℝ{3}(1.5, -1.5, 25.0), 1/5000, 10.0
tt = 0.0:dt:T
XX, _ = simulateSegment(ℝ{3}(0.0, 0.0, 0.0), x0, Pˣ, tt)


θ₀ = [5.0, 15.0, 6.0, 8.0]
Pˣ = LorenzCV(θ₀...)


skip = 1000

Σdiagel = 10^0
Σ = SMatrix{2,2}(1.0I)*Σdiagel
L = @SMatrix[1.0 0.0 0.0;
             0.0 1.0 0.0]

obs_time, obs_vals = XX.tt[1:skip:end], [rand(Gaussian(L*x, Σ)) for x in XX.yy[1:skip:end]]

aux_flag = Val{(true,true,false)}()
P̃ = [LorenzCVAux(θ₀..., t₀, u, T, v, aux_flag, x0[3]) for (t₀, T, u, v)
     in zip(obs_time[1:end-1], obs_time[2:end], obs_vals[1:end-1], obs_vals[2:end])]

setup = MCMCSetup(Pˣ, P̃, PartObs())
set_observations!(setup, [L for _ in P̃], [Σ for _ in P̃], obs_vals, obs_times) # uses default fpt
set_imputation_grid!(setup,
                     1/2000,                                               # dt
                     (t₀,T) -> ( (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀)) ) # time transformation
                     )
set_transition_kernels!(setup,
                        RandomWalk([2.0, 1.0, 0.64, 0.3],
                                   [false, false, false, true]), # transition kernel
                        0.995,                       # precond. Crank-Nicolson memory parameter
                        true,                        # whether to update parameters
                        (Val((true, false, false, false)),
                         Val((false, true, false, false)),
                         Val((false, false, true, false)),
                         Val((false, false, false, true))
                        ),                           # coordinate updates
                        (MetropolisHastingsUpdt(),
                         MetropolisHastingsUpdt(),
                         MetropolisHastingsUpdt(),
                         MetropolisHastingsUpdt()
                        ),                           # update types
                        NoAdaptation()               # adaptation of guid prop
                        )
set_priors!(setup,
            Priors((ImproperPrior(),
                    ImproperPrior(),
                    ImproperPrior(),
                    ImproperPrior())),               # priors over parameters
            GsnStartingPt(x0, x0, @SMatrix [20.0 0.0 0.0;
                                            0.0 20.0 0.0;
                                            0.0 0.0 400.0]) # prior over starting point
            )
set_mcmc_params!(setup,
                 1*10^4,            # number of mcmc steps
                 1*10^3,            # save path every ... iteration
                 10^2,              # print progress message every ... iteration
                 10^0,              # thin the path imputatation points for save
                 100                # number of first iterations without param update
                 )
set_blocking!(setup)    # use default no blocking setting
set_solver!(setup, Vern7(), NoChangePt())

Random.seed!(4)
#(chain, accRateImp, accRateUpdt,
#    paths, time_)
start = time()
mcmc_results = mcmc(eltype(x0), setup)
elapsed = time() - start
print("time elapsed: ", elapsed, "\n")


print("imputation acceptance rate: ", accRateImp,
      ", parameter update acceptance rate: ", accRateUpdt)

using Plots
pTp = [[[x[i] for x in path] for path in paths] for i in 1:3]

function plotPaths(j, obsIdxS, obsIdxE, show_obs=true)
    idxS = div((obsIdxS-1)*skip,5)+1
    idxE = div((obsIdxE-1)*skip,5)+1
    p = plot()
    for i in 1:length(paths)
        plot!(time_[idxS:idxE], pTp[j][i][idxS:idxE], label="", color="steelblue", alpha=0.2, linewidth=0.2)
    end
    if show_obs
        scatter!(obsTime[obsIdxS:obsIdxE], [x[j] for x in obsVals][obsIdxS:obsIdxE],
                 color="orange", label="")
    end
    p
end

plotPaths(1, 1, 10)
plotPaths(2, 1, 10)
plotPaths(3, 1, 10, false)

plot([θ[1] for θ in chain])
plot([θ[2] for θ in chain])
plot([θ[3] for θ in chain])
plot([θ[4] for θ in chain])
