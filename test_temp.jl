mkpath("output/")
outdir="output"


using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models: ℝ
using DataFrames
using CSV

L = @SMatrix [1.0]
Σ = @SMatrix[10^(-6)]

include("src/sinDiffusion.jl")
include("src/types.jl")
include("src/ralston3.jl")
include("src/rk4.jl")
include("src/tsit5.jl")
include("src/vern7.jl")

include("src/guid_prop_bridge.jl")
include("src/random_walk.jl")
include("src/mcmc.jl")
include("src/temperature_mcmc.jl")

include("src/save_to_files.jl")

x0 = ℝ{1}(0.0)
obs = ℝ{1}.([0.0, 0.5*π])
obsTime = [0.0, 8.0]
fpt = [NaN]
fptOrPartObs = PartObs()
θ₀ = [2.0, -2.0, 8.0, 0.5]
P˟ = SinDiffusion(θ₀...)
P̃ = [SinDiffusionAux(θ₀..., t₀, u[1], T, v[1]) for (t₀, T, u, v) in
        zip(obsTime[1:end-1], obsTime[2:end], obs[1:end-1], obs[2:end])]
Ls = [L for _ in P̃]
Σs = [Σ for _ in P̃]
τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
numSteps=10^4
tKernel = RandomWalk([1.0, 1.0, 1.0, 1.0], [false, false, false, true])
priors = ((ImproperPrior(),),)
Random.seed!(4)
(chain, accRateImp, accRateUpdt,
    paths, time_) = mcmc(fptOrPartObs, obs, obsTime, x0, 0.0, P˟, P̃, Ls, Σs,
                         numSteps, tKernel, priors, τ;
                         fpt=fpt,
                         ρ=0.9,
                         dt=1/5000,
                         saveIter=3*10^2,
                         verbIter=10^2,
                         updtCoord=(Val((true, true, true, false, false)),
                                    #Val((true, false, false, false, false)),
                                    #Val((false, true, false, false, false)),
                                    #Val((false, false, true, false, false)),
                                    Val((false, false, false, false, true)),
                                    ),
                         paramUpdt=false,
                         updtType=(MetropolisHastingsUpdt(),
                                   #MetropolisHastingsUpdt(),
                                   #MetropolisHastingsUpdt(),
                                   MetropolisHastingsUpdt(),
                                   ),
                         skipForSave=10^1,
                         solver=Vern7())

print("imputation acceptance rate: ", accRateImp,
      ", parameter update acceptance rate: ", accRateUpdt)


df2 = savePathsToFile(paths, time_, joinpath(outdir, "sampled_paths.csv"))
df3 = saveChainToFile(chain, joinpath(outdir, "chain.csv"))

include("src/plots.jl")
# make some plots
set_default_plot_size(30cm, 20cm)

plot(df2, x=:time, y=:x1, color=:idx, Geom.line,
     Scale.color_continuous(colormap=Scale.lab_gradient("#fceabb", "#a2acae",
                                                        "#36729e")))

plotChain(df3, coords=[1])
plotChain(df3, coords=[2])
plotChain(df3, coords=[3])
plotChain(df3, coords=[4])
