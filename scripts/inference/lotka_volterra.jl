SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)


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

θˣ = [1.5, 1.0, 3.0, 1.0, 0.1, 0.1]
Pˣ = LotkaVolterraDiffusion(θˣ...)

x0, dt, T = ℝ{2}(2.0, 2.0), 1/5000, 10.0
tt = 0.0:dt:T

XX, _ = simulate_segment(ℝ{2}(0.0, 0.0), x0, Pˣ, tt)


θ_init = copy(θˣ)
Pˣ = LotkaVolterraDiffusion(θ_init...)

skip = 2000

Σdiagel = 10^0
Σ = @SMatrix[Σdiagel]#SMatrix{2,2}(1.0I)*Σdiagel
L = @SMatrix[1.0 0.0]

obs_time, obs_vals = XX.tt[1:skip:end], [rand(Gaussian(L*x, Σ)) for x in XX.yy[1:skip:end]]

P̃ = [LotkaVolterraDiffusionAux(θ_init..., t₀, u, T, v) for (t₀, T, u, v)
     in zip(obs_time[1:end-1], obs_time[2:end], obs_vals[1:end-1], obs_vals[2:end])]

model_setup = DiffusionSetup(Pˣ, P̃, PartObs())
set_observations!(model_setup, [L for _ in P̃], [Σ for _ in P̃], obs_vals, obs_time) # uses default fpt
set_imputation_grid!(model_setup, 1/1000)
set_x0_prior!(model_setup,
              GsnStartingPt(x0, @SMatrix [20.0 0.0;
                                          0.0 20.0;]),
              x0)
set_auxiliary!(model_setup; skip_for_save=10^0,
               adaptive_prop=NoAdaptation())
initialise!(eltype(x0), model_setup, Vern7(), false, NoChangePt(100))

readj = (100, 0.001, 0.001, 999.9, 0.234, 50)

mcmc_setup = MCMCSetup(
      Imputation(NoBlocking(), 0.99, Vern7()),
      ParamUpdate(ConjugateUpdt(), [1,2,3,4], θ_init, nothing,
                  MvNormal(fill(0.0, 4), diagm(0=>fill(1000.0, 4))),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, [1,2,3,4]))),
      ParamUpdate(MetropolisHastingsUpdt(), 1, θ_init,
                  UniformRandomWalk(0.1, true), ImproperPrior(),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, 1)), readj),
      ParamUpdate(MetropolisHastingsUpdt(), 2, θ_init,
                  UniformRandomWalk(0.1, true), ImproperPrior(),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, 2)), readj),
      ParamUpdate(MetropolisHastingsUpdt(), 3, θ_init,
                  UniformRandomWalk(0.1, true), ImproperPrior(),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, 3)), readj),
      ParamUpdate(MetropolisHastingsUpdt(), 4, θ_init,
                  UniformRandomWalk(0.1, true), ImproperPrior(),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, 4)), readj),
      ParamUpdate(MetropolisHastingsUpdt(), 5, θ_init,
                  UniformRandomWalk(0.1, true), ImproperPosPrior(),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, 5)), readj),
      ParamUpdate(MetropolisHastingsUpdt(), 6, θ_init,
                  UniformRandomWalk(0.1, true), ImproperPosPrior(),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, 6)), readj))

schedule = MCMCSchedule(5*10^4, [[1,3,4,5]],
                        (save=1*10^3, verbose=10^2, warm_up=100,
                         readjust=(x->x%100==0), fuse=(x->false)))

Random.seed!(4)
out = mcmc(mcmc_setup, schedule, model_setup)

include(joinpath(SRC_DIR, DIR, "plotting_fns.jl"))
plot_chains(out[2]; truth=θˣ)
#=
plot_paths(out[1], out[2], schedule; obs=(times=obs_time[2:end],
                     vals=[[v[1] for v in obs_vals[2:end]],
                           [v[2] for v in obs_vals[2:end]]], indices=[2,3]))
=#
