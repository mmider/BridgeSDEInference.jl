SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using BridgeSDEInference
include(joinpath(SRC_DIR, "BridgeSDEInference_for_tests.jl"))

#const BSI = BridgeSDEInference
using DataFrames, DelimitedFiles, CSV
using Test

using Bridge, StaticArrays, Distributions #BridgeSDEInference
using Statistics, Random, LinearAlgebra
using GaussianDistributions

DIR = "auxiliary"
include(joinpath(SRC_DIR, DIR, "read_and_write_data.jl"))
include(joinpath(SRC_DIR, DIR, "transforms.jl"))
const 𝕏 = SVector
# decide if first passage time observations or partially observed diffusion
fptObsFlag = false

# pick dataset
using DelimitedFiles
using Makie
#data = readdlm("../LinneasData190920.csv", ';')
#
#data[isnan.(data)] .= circshift(data, (-1,0))[isnan.(data)]
#data[isnan.(data)] .= circshift(data, (1,0))[isnan.(data)]
#data[isnan.(data)] .= circshift(data, (-2,0))[isnan.(data)]
#data[isnan.(data)] .= circshift(data, (2,0))[isnan.(data)]
#data[isnan.(data)] .= circshift(data, (3,0))[isnan.(data)]
#any(isnan.(data))
# t = 0:30:N
#data = cumsum(0.1rand(200,100), dims=1) # Mock data

#N, K = size(data)

#x0 = [𝕏(x, 0.0) for x in data[:, 1]]
#obs = map(𝕏, data)
#obs_times = hcat([range(0, 1, length=N) for k in 1:K]...)

𝕂 = Float64
L = @SMatrix [1. 0.]
Σdiagel = 0.3^2
Σ = @SMatrix [Σdiagel]
Noise = Gaussian(𝕏(0.0), Σ)
sim = [:simulate, :linnea][1]
if sim == :simulate
      include(joinpath("..", "data_generation", "simulate_repeated_fhn.jl"))
      K = length(XX)
      obs = [map(x->L*x + rand(Noise), XX[k].yy) for k in 1:K]
      obs_times = [XX[k].tt for k in 1:K]
      n = (0.5+rand(), 0.5 + rand(), 0.5 + rand())
      θ₀ = (10.0n[1], -8.0n[2], 15.0n[3], 0.0, 3.0)
      dt = 1/1000

elseif sim == :linnea

      data = readdlm("../LinneasData190920.csv", ';')
      data = data[1:end, 1:end÷3]
      _, K = size(data)
      y = data[:, 1]
      st = 0.01
      t = range(0.0, length=length(y), step = st)
      s = .!isnan.(y)
      X = SamplePath(t[s], y[s])
      XX = [X]
      for k in 2:K
            y = data[:, k]
            t = range(0.0, length=length(y), step = st)
            s = .!isnan.(y)
            X = SamplePath(t[s], y[s])
            push!(XX, X)
      end
      K = length(XX)
      obs = [map(y->2*(𝕏(y)-0.5), XX[k].yy) for k in 1:K]
      obs_times = [XX[k].tt for k in 1:K]
      x0 = [𝕏(X.yy[1], 0.0) for X in XX]
      θ₀ = (10.0, -8.0, 15.0, 0.0, 0.3*10)
      dt = 1/100
      # 0.0993388433093311, -0.46293689190595844, 1.307687515228687
end


param = :complexConjug
# Target law
P˟ = FitzhughDiffusion(param, θ₀...)

P̃ = map(1:K) do k
      map(1:length(obs[k])-1) do i
            t₀, T, u, v = obs_times[k][i], obs_times[k][i+1], obs[k][i], obs[k][i+1]
            FitzhughDiffusionAux(param, θ₀..., t₀, u[1], T, v[1])
      end
end

Ls = fill.(Ref(L), length.(XX))
Σs = fill.(Ref(Σ), length.(XX))



setups = [MCMCSetup(P˟, P̃[k], PartObs()) for k in 1:K]
set_observations!.(setups, Ls, Σs, obs, obs_times)
for k in 1:K set_imputation_grid!(setups[k], dt) end

t_kernel = RandomWalk([3.0, 5.0, 5.0, 0.01, 0.5], 5)

#NOTE there is quite a bit of redundancy here, will need to be adjusted with
# a more tailored `setup`
for k in 1:K
    set_transition_kernels!(setups[k],
                            [RandomWalk([],[]), t_kernel],
                            0.975, true, ((1,2,3), (5,)),
                            (ConjugateUpdt(),
                             #MetropolisHastingsUpdt(),
                             MetropolisHastingsUpdt(),
                             ))
    set_priors!(setups[k],
                Priors((MvNormal([0.0,0.0,0.0], diagm(0=>[1000.0, 1000.0, 1000.0])),
                       ImproperPrior(),
                       #ImproperPrior(),
                       )),
                GsnStartingPt(𝕏(obs[k][1][1], 0.0), @SMatrix [20. 0; 0 20.]),
                𝕏(obs[k][1][1], -4rand()))
end
setups
for k in 1:K set_mcmc_params!(setups[k], 5*10^3, 3*10^2, 10^2, 10^0, 100) end
for k in 1:K set_solver!(setups[k], Vern7(), NoChangePt()) end
for k in 1:K initialise!(𝕂, setups[k]) end

Random.seed!(4)
out, elapsed = mcmc(𝕂, setup)
display(out.accpt_tracker)

#=
x0⁺, pathsToSave = transformMCMCOutput(x0, paths, saveIter; chain=chain,
                                       numGibbsSteps=2,
                                       parametrisation=param,
                                       warmUp=warmUp)
#using Makie

df2 = savePathsToFile(pathsToSave, time_, joinpath(OUT_DIR, "sampled_paths.csv"))
=#

df3 = saveChainToFile(chain, joinpath(OUT_DIR, "chain.csv"))

#=
include(joinpath("..","src","auxiliary","plotting_fns.jl"))
set_default_plot_size(30cm, 20cm)
plotPaths(df2, obs=[Float64.(df.x1), [x0⁺[2]]],
          obs_times=[Float64.(df.time), [0.0]], obsCoords=[1,2])

plotChain(df3, coords=[1])
plotChain(df3, coords=[2])
plotChain(df3, coords=[3])
plotChain(df3, coords=[5])
=#

p1 = lines(df3[!,1])
lines!(df3[!,2])
lines!(df3[!,3])
lines!(df3[!,end])


θ̂ = mean(chain[end÷2:end])
P̂ = BSI.clone(P˟, θ̂)

p1
