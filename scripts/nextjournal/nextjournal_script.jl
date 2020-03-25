## Intro
using Bridge
using StaticArrays
using BridgeSDEInference
using Random, LinearAlgebra, Distributions
const State = SArray{Tuple{2},T,1,2} where {T};

param = :regular # regular parametrization of the trajectories (no transformation of the coordinate processes)
ε = 0.1 ; s =-0.8 ; γ =1.5 ; β = 0.0 ; σ =0.3 ;
P = FitzhughDiffusion(param, ε, s, γ, β, σ);

include(joinpath(pathof(BridgeSDEInference), "..", "auxiliary", "data_simulation_fns.jl"))
# starting point under :regular parametrisation
x0 = State(-0.5, -0.6)

## Simulation
# time grid
dt = 1/50000
T = 20.0
tt = 0.0:dt:T

Random.seed!(4)
X, _ = simulate_segment(0.0, x0, P, tt);

# subsampling
num_obs = 100
skip = div(length(tt), num_obs)
Σ = [10^(-4)]
L = [1.0 0.0]
obs = (time = X.tt[1:skip:end],
values = [Float64(rand(MvNormal(L*x, Σ))[1]) for x in X.yy[1:skip:end]]);

## Plotting
using Plots
gr()
Plots.plot(X.tt,  first.(X.yy), label = "X")
Plots.plot!(X.tt,  last.(X.yy), label = "Y")
Plots.scatter!(obs.time, obs.values, markersize=1.5, label = "observations")

## Inference
θ_init = [ε, s, γ, β, σ].*(1 .+ randn(5).*[0.5, 0.5, 0.5, 0.0, 0.5])
# Verify that the parameters lay in their domain
θ_init[1]<0 ? θ_init[1] = -θ_init[1] : θ_init[1] = θ_init[1]
θ_init[3]<0 ? θ_init[3] = -θ_init[3]  : θ_init[3] = θ_init[3]
θ_init[5]<0 ? θ_init[5] = -θ_init[5] : θ_init[5] = θ_init[5];
# Take the real β, as it is fixed.

P_trgt = FitzhughDiffusion(param, θ_init...)
P_aux = [FitzhughDiffusionAux(param, θ_init..., t₀, u, T, v) for (t₀,T,u,v)
        in zip(obs.time[1:end-1], obs.time[2:end], obs.values[1:end-1], obs.values[2:end])]

# Container
model_setup = DiffusionSetup(P_trgt, P_aux, PartObs());

initialise!(eltype(x0), model_setup, Vern7(), false, NoChangePt(100))
# Further setting
set_auxiliary!(model_setup; skip_for_save=1, adaptive_prop=NoAdaptation());

mcmc_setup = MCMCSetup(
      Imputation(NoBlocking(), 0.975, Vern7()),
      ParamUpdate(MetropolisHastingsUpdt(), 1, θ_init,
                  UniformRandomWalk(0.5, true), ImproperPosPrior(),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P_aux, 1))
                  ),
      ParamUpdate(MetropolisHastingsUpdt(), 2, θ_init,
                  UniformRandomWalk(0.5, false), ImproperPrior(),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P_aux, 2))
                  ),
      ParamUpdate(MetropolisHastingsUpdt(), 3, θ_init,
                  UniformRandomWalk(0.5, true), ImproperPosPrior(),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P_aux, 3))
                  ),
      ParamUpdate(MetropolisHastingsUpdt(), 5, θ_init,
                  UniformRandomWalk(0.5, true), ImproperPosPrior(),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P_aux, 5))
                  ))

schedule = MCMCSchedule(1*10^3, [[1,2,3,4,5]],
                        (save=3*10^2, verbose=10^2, warm_up=100,
                         readjust=(x->x%100==0), fuse=(x->false)));

Random.seed!(4)
out = mcmc(mcmc_setup, schedule, model_setup);

                         
