SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
include(joinpath(SRC_DIR, "BridgeSDEInference_for_tests.jl"))


using StaticArrays
using Distributions
using Random
# Let's generate the data
# -----------------------
using Bridge
DIR = "auxiliary"
include(joinpath(SRC_DIR, DIR, "data_simulation_fns.jl"))
include(joinpath(SRC_DIR, DIR, "utility_functions.jl"))
Random.seed!(4)
#θ₀ = [10.0, 28.0, 8.0/3.0, 3.0, 3.0, 3.0]
θˣ = [10.0, 28.0, 8.0/3.0, 3.0]
Pˣ = LorenzCV(θˣ...)

x0, dt, T = ℝ{3}(1.5, -1.5, 25.0), 1/5000, 4.0
tt = 0.0:dt:T
XX, _ = simulateSegment(ℝ{3}(0.0, 0.0, 0.0), x0, Pˣ, tt)


θ_init = [5.0, 15.0, 6.0, 8.0]
Pˣ = LorenzCV(θ_init...)


skip = 1000

Σdiagel = 10^0
Σ = SMatrix{2,2}(1.0I)*Σdiagel
L = @SMatrix[0.0 1.0 0.0;
             0.0 0.0 1.0]

obs_time, obs_vals = XX.tt[1:skip:end], [rand(Gaussian(L*x, Σ)) for x in XX.yy[1:skip:end]]

aux_flag = Val{(false,true,true)}()
P̃ = [LorenzCVAux(θ_init..., t₀, u, T, v, aux_flag, x0[3]) for (t₀, T, u, v)
     in zip(obs_time[1:end-1], obs_time[2:end], obs_vals[1:end-1], obs_vals[2:end])]

setup = MCMCSetup(Pˣ, P̃, PartObs())
set_observations!(setup, [L for _ in P̃], [Σ for _ in P̃], obs_vals, obs_time) # uses default fpt
set_imputation_grid!(setup, 1/2000)
set_transition_kernels!(setup,
                        [RandomWalk([], []),
                         RandomWalk([2.0, 1.0, 0.64, 0.3], 4)],
                        0.96, true, [[1,2,3],[4]],
                        (ConjugateUpdt(),
                         MetropolisHastingsUpdt()
                        ),                           # update types
                        Adaptation(x0,
                                   [0.7, 0.4, 0.2, 0.2, 0.2],
                                   [500, 500, 500, 500, 500],
                                   1)
                        )
set_priors!(setup,
            Priors((MvNormal([0.0,0.0,0.0], diagm(0=>[1000.0, 1000.0, 1000.0])),
                    ImproperPrior())),               # priors over parameters
            GsnStartingPt(x0, @SMatrix [20.0 0.0 0.0;
                                        0.0 20.0 0.0;
                                        0.0 0.0 400.0]), # prior over starting point
            x0
            )
set_mcmc_params!(setup,
                 4*10^3,            # number of mcmc steps
                 1*10^3,            # save path every ... iteration
                 10^2,              # print progress message every ... iteration
                 10^0,              # thin the path imputatation points for save
                 100                # number of first iterations without param update
                 )
set_blocking!(setup, ChequeredBlocking(),
              (collect(1:length(obs_vals)-2)[1:2:end], 10^(-10), SimpleChangePt(100)))
set_solver!(setup, Vern7(), NoChangePt())
initialise!(eltype(x0), setup)


Random.seed!(4)
out, elapsed = @timeit mcmc(setup)
display(out.accpt_tracker)

include(joinpath(SRC_DIR, DIR, "plotting_fns.jl"))
plot_chains(out; truth=[10.0, 28.0, 8.0/3.0, 3.0],
            ylims=[nothing, (25,30), (2,5), (0,10)])
plot_paths(out; obs=(times=obs_time[2:end],
                     vals=[[v[1] for v in obs_vals[2:end]],
                           [v[2] for v in obs_vals[2:end]]], indices=[2,3]))
