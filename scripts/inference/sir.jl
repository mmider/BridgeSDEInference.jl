SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)
using Makie: lines, lines!

include(joinpath(SRC_DIR, "BridgeSDEInference_for_tests.jl"))

using StaticArrays
using Distributions
using Random
# Let's generate the data
# -----------------------
using Bridge
#import BridgeSDEInference: CIR, CIRaux
DIR = "auxiliary"
include(joinpath(SRC_DIR, DIR, "data_simulation_fns.jl"))
include(joinpath(SRC_DIR, DIR, "utility_functions.jl"))
Random.seed!(4)
pop = 50_000_000
K = 1.0
θˣ = [0.37, 0.05, 0.05, 0.01, K]

Pˣ = SIR(θˣ...)

x0, dt, T = ℝ{2}(1/pop, 0.), 1/5000, 30.0
tt = 0.0:dt:T

XX, _ = simulate_segment(ℝ{2}(1.0, 0.0), x0, Pˣ, tt)
last(XX)[2]*pop

#lines(XX.tt, K .- sum.(XX.yy))
lines(XX.tt, pop*first.(XX.yy))
lines!(XX.tt, pop*last.(XX.yy))

θ_init = copy(θˣ)
Pˣ = SIR(θ_init...)

length(XX.tt)
skip = 20000

#Σdiagel =
Σ = @SMatrix[5.0 0.0; 0.0 1.0]/pop
L = @SMatrix[1.0 0.0; 0.0 1.0]

obs_time, obs_vals = XX.tt[1:skip:end], [rand(Gaussian(L*x, Σ)) for x in XX.yy[1:skip:end]]

days = [0.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
cases = [1.0 0.0; 62.0 0.0; 121.0 0.1; 198.0 3.0; 291.0 6.0; 440.0 9.0; 571.0 17.0; 830.0 25.0; 1287.0 41.0; 1975.0 56.0; 2744.0 80.0; 4515.0 106.0; 5974.0 132.0; 7771.0 170.0]
if true
      obs_time = days
      obs_vals = (1/pop)*reinterpret(SVector{2,Float64}, (cases)')
end

P̃ = [SIRAux(θ_init..., t₀, u, T, v) for (t₀, T, u, v)
     in zip(obs_time[1:end-1], obs_time[2:end], obs_vals[1:end-1], obs_vals[2:end])]

model_setup = DiffusionSetup(Pˣ, P̃, PartObs())
set_observations!(model_setup, [L for _ in P̃], [Σ for _ in P̃], obs_vals, obs_time) # uses default fpt
set_imputation_grid!(model_setup, 1/1000)
set_x0_prior!(model_setup,
              GsnStartingPt(x0, @SMatrix [0.01 0.0;
                                          0.0 0.01;]),
              x0)
set_auxiliary!(model_setup; skip_for_save=10^0,
               adaptive_prop=NoAdaptation())
initialise!(eltype(x0), model_setup, Vern7(), false, NoChangePt(100))
#:step, :scale, :min, :max, :trgt, :offset
readj = (100, 0.001, 0.001, 999.9, 0.4, 50)

mcmc_setup = MCMCSetup(
      Imputation(NoBlocking(), 0.99, Vern7()),
      ParamUpdate(ConjugateUpdt(), [1,2], θ_init, nothing,
                  MvNormal(fill(0.0, 2), diagm(0=>fill(1000.0, 2))),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, [1,2]))),
      ParamUpdate(MetropolisHastingsUpdt(), 1, θ_init,
                  UniformRandomWalk(0.1, true), ImproperPrior(),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, 1)), readj),
      ParamUpdate(MetropolisHastingsUpdt(), 2, θ_init,
                  UniformRandomWalk(0.01, true), ImproperPrior(),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, 2)), readj),
      ParamUpdate(MetropolisHastingsUpdt(), 3, θ_init,
                  UniformRandomWalk(0.1, true), ImproperPrior(),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, 3)), readj),
      ParamUpdate(MetropolisHastingsUpdt(), 4, θ_init,
                  UniformRandomWalk(0.1, true), ImproperPrior(),
                  UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, 4)), readj),
      )

schedule = MCMCSchedule(2*10^4, [[1],[2]], #[[1],[2], [5]],
                        (save=1*10^3, verbose=10^2, warm_up=100,
                         readjust=(x->x%100==0), fuse=(x->false)))

Random.seed!(4)
out = mcmc(mcmc_setup, schedule, model_setup)
ws = out[2]

θs = ws.θ_chain
±(a, b) = a - b, a + b
beta, gamma, s1 =  [median(getindex.(θs, i)) for i in [1,2,3]] .± [std(getindex.(θs, i)) for i in [1,2, 3]]
R0 = mean(getindex.(θs, 1)./getindex.(θs, 2)) ± std(getindex.(θs, 1)./getindex.(θs, 2))

@show beta
@show gamma
@show s1
@show R0

lines(getindex.(θs, 1))
lines(getindex.(θs, 2))
lines(getindex.(θs, 3))

include(joinpath(SRC_DIR, DIR, "plotting_fns.jl"))
plot_chains(ws; truth=θˣ)
#=
plot_paths(out[1], out[2], schedule; obs=(times=obs_time[2:end],
                     vals=[[v[1] for v in obs_vals[2:end]],
                           [v[2] for v in obs_vals[2:end]]], indices=[2,3]))
=#
