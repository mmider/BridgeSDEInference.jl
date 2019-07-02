#NOTE see README for explanations

mkpath("output/")
outdir="output"

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using DataFrames
using CSV
using ForwardDiff: value
const ℝ = SVector{N,T} where {N,T}
# specify observation scheme
L = @SMatrix [1.0]
Σ = @SMatrix [10^(-6)]

include("src/sin_diffusion.jl")

#NOTE important! MCMCBridge must be imported after FHN is loaded
#include("src/MCMCBridge.jl")
#using Main.MCMCBridge
include("src/types.jl")
include("src/ralston3.jl")
include("src/rk4.jl")
include("src/tsit5.jl")
include("src/vern7.jl")

include("src/priors.jl")

include("src/guid_prop_bridge.jl")
include("src/random_walk.jl")
include("src/mcmc.jl")

include("src/save_to_files.jl")

x0 = ℝ{1}(0.0)
fptOrPartObs = PartObs()

θ₀ = [4.0, -4.0, 4.0, 1.0]
# Target law
P˟ = SinDiffusion(θ₀...)


obs = ℝ{1}.([0.0, 2*π])
obsTime = [0.0, 4.0]

# Auxiliary law
P̃ = [SinDiffusionAux(θ₀..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
     in zip(obsTime[1:end-1], obsTime[2:end], obs[1:end-1], obs[2:end])]

fpt = [NaN for _ in P̃]
Ls = [L for _ in P̃]
Σs = [Σ for _ in P̃]
τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
numSteps=3*10^4
tKernel = RandomWalk([0.015, 5.0, 0.05, .0],
                     [false, false, false, true])
priors = Priors((ImproperPrior(),
                 #ImproperPrior()
                 )
                #(ImproperPrior(),)
                )

Random.seed!(4)
(chain, accRateImp, accRateUpdt,
    paths, time_) = mcmc(eltype(x0), fptOrPartObs, obs, obsTime, x0, 0.0, P˟, P̃, Ls, Σs,
                         numSteps, tKernel, priors, τ;
                         fpt=fpt,
                         ρ=0.0,
                         dt=1/5000,
                         saveIter=3*10^2,
                         verbIter=10^2,
                         updtCoord=(Val((true, false, false)),
                                    #Val((true, false, false, false, false)),
                                    #Val((false, true, false, false, false)),
                                    #Val((false, false, true, false, false)),
                                    #Val((true, false, false, false, false)),
                                    ),
                         paramUpdt=false,
                         updtType=(#ConjugateUpdt(),
                                   #MetropolisHastingsUpdt(),
                                   #MetropolisHastingsUpdt(),
                                   MetropolisHastingsUpdt(),
                                   #MetropolisHastingsUpdt(),
                                   ),
                         skipForSave=10^1,
                         solver=Vern7())

print("imputation acceptance rate: ", accRateImp,
      ", parameter update acceptance rate: ", accRateUpdt)

pathsToSave = paths
#df2 = savePathsToFile(pathsToSave, time_, joinpath(outdir, "sampled_paths.csv"))
#df3 = saveChainToFile(chain, joinpath(outdir, "chain.csv"))

df2 = savePathsToFile(paths, time_, joinpath(outdir, "sampled_paths_sin_example_comp.csv"))
include("src/plots.jl")
# make some plots
set_default_plot_size(30cm, 20cm)
plotPaths(df2, obs=[x0], obsTime=[[0.0]], obsCoords=[1])
