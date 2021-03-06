using LinearAlgebra, Distributions

param = :regular
θ_init = [0.1, -0.8, 1.5, 0.0, 0.3]
P_trgt = FitzhughDiffusion(param, θ_init...)

P_aux = [FitzhughDiffusionAux(param, θ_init..., t₀, u, T, v) for (t₀,T,u,v)
        in zip(obs.time[1:end-1], obs.time[2:end], obs.values[1:end-1], obs.values[2:end])]


L = @SMatrix [1. 0.]
Σ = @SMatrix [10^(-3)]

model_setup = DiffusionSetup(P_trgt, P_aux, PartObs())
set_observations!(model_setup, [L for _ in P_aux], [Σ for _ in P_aux],
                  obs.values, obs.time)
set_imputation_grid!(model_setup, 1/400)
set_x0_prior!(model_setup, KnownStartingPt(x0))
initialise!(eltype(x0), model_setup, Vern7(), false, NoChangePt(100))
set_auxiliary!(model_setup; skip_for_save=10^0, adaptive_prop=NoAdaptation())

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

schedule = MCMCSchedule(1*10^4, [[1,2,3,4,5]],
                        (save=3*10^2, verbose=10^2, warm_up=100,
                         readjust=(x->x%100==0), fuse=(x->false)))

Random.seed!(4)
out = mcmc(mcmc_setup, schedule, model_setup)





using PyPlot
plot(out[1].time, out[1].paths[10])
scatter(obs...)


out[1].time
out[1].paths[3]


for j in [1,2,3,5]
    plot([out[2].θ_chain[i][j] for i in 1:length(out[2].θ_chain)])
end
cred_interval(x, p = 0.95) = quantile(x, [(1-p)/2, 1-(1-p)/2])
[cred_interval(getindex.(out[2].θ_chain, j)) for j in [1,2,3,5]]
