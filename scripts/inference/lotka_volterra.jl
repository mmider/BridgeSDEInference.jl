using StaticArrays
using Distributions
using Random
using Bridge
using Statistics, LinearAlgebra
using GaussianDistributions
using BridgeSDEInference
const SDE = BridgeSDEInference


## Helper functions and directories

SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)

DIR = "auxiliary"
include(joinpath(SRC_DIR, DIR, "data_simulation_fns.jl"))
include(joinpath(SRC_DIR, DIR, "utility_functions.jl"))


### Import example SDE LotkaVolterraDiffusion

using BridgeSDEInference: LotkaVolterraDiffusion, LotkaVolterraDiffusionAux

#=
Stochastic Lotka-Volterra System (SDE) with
parameters (α, β, γ, δ, σ1, σ2) with drift

     b(x) = [α*x[1] - β*x[1]*x[2],
             δ*x[1]*x[2] - γ*x[2]]

and volatility

    σ(x) = [ σ1 0
             0  σ2 ]

# parameters: see src/examples/lotka_volterra.jl
=#


### Generate data with parameters

Random.seed!(4)
θˣ = [1.5, 1.0, 3.0, 1.0, 0.2, 0.2] #  (α, β, γ, δ, σ1, σ2)
Pˣ = LotkaVolterraDiffusion(θˣ...)

x0 = ℝ{2}(2.5, 2.) # starting point
w0 = ℝ{2}(0.0, 0.0) # starting point of driving Wiener process

dt, T =  1/5000, 20.0 # time grid
tt = 0.0:dt:T

XX, _ = simulate_segment(w0, x0, Pˣ, tt) # simulate process


### Subsampling observations

skip = 2000

# Set observation scheme: partial observations of first coordinate
# with observation operator L and Gaussian noise
Σdiagel = 1.0
Σ = @SMatrix[Σdiagel]
L = @SMatrix[1.0 0.0]

# Observe with noise at sample times
obs_time, obs_vals = XX.tt[1:skip:end], [rand(Gaussian(L*x, Σ)) for x in XX.yy[1:skip:end]]


### Setup MCMC scheme to infer parameters given observations

θ_init = copy(θˣ)
Pˣ = LotkaVolterraDiffusion(θ_init...)

# Setup: For observation intervals define linearizations of the diffusion to be used
# in generating MCMC proposals
P̃ = [LotkaVolterraDiffusionAux(θ_init..., t₀, u, T, v) for (t₀, T, u, v)
     in zip(obs_time[1:end-1], obs_time[2:end], obs_vals[1:end-1], obs_vals[2:end])]

# Setup: Partial observations: register observation scheme of each observation
model_setup = DiffusionSetup(Pˣ, P̃, PartObs())
set_observations!(model_setup, [L for _ in P̃], [Σ for _ in P̃], obs_vals, obs_time)
set_imputation_grid!(model_setup, 1/1000)

# Setup: Priors
set_x0_prior!(model_setup,
              GsnStartingPt(x0, @SMatrix [20.0 0.0;
                                          0.0 20.0;]),
              x0)

# Initialize sampler
set_auxiliary!(model_setup; skip_for_save=10^0,
               adaptive_prop=NoAdaptation())
initialise!(eltype(x0), model_setup, Vern7(), false, NoChangePt(100))

# Parameters for adaptively finding good Metropolis stepsize
readj = (100, 0.001, 0.001, 999.9, 0.234, 50)

# Finalize setup:
# Define MCMC steps for 1.) path imputation, 2.) conjugate updates for
# linear parameters α, β, γ, δ and 3.) - 8.) random walk steps
# for parameters α, β, γ, δ, σ1, σ2 (in that order)
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

# Set up schedule for MCMC: Impute, and estimate parameters α, β, and δ (Steps 1, 3, 4, 6)
schedule = MCMCSchedule(1*10^3, [[1],[3,4,6]], # which (groups of) steps in which order
                        (save=1*10^3, verbose=10^2, warm_up=100,
                         readjust=(x->x%100==0), fuse=(x->false)))


### Run MCMC

Random.seed!(4)
state, history = mcmc(mcmc_setup, schedule, model_setup)

# most importantly: parameter samples
θs = history.θ_chain

# Compare posterior mean with true parameters, for example
@show [mean(θs) θˣ]

### Post processing: plot results

include(joinpath(SRC_DIR, DIR, "plotting_fns.jl"))
plot_chains(history; truth=θˣ)

plot_paths(state, history, schedule; obs=(times=obs_time[2:end],
                     vals=[[v[1] for v in obs_vals[2:end]]], indices=[1]))
