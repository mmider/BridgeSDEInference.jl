#NOTE see README for explanations

mkpath("output/")
outdir="output"

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models: ℝ
using DataFrames
using CSV

# specify observation scheme
L = @SMatrix [1.0]
Σdiagel = 10^(-5)
Σ = @SMatrix [Σdiagel]



#NOTE important! MCMCBridge must be imported after FHN is loaded
#include("src/MCMCBridge.jl")
#using Main.MCMCBridge

include("src/sinDiffusion.jl")
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
θ₀ = [2.0, -2.0, 8.0, 0.5]
P˟ = SinDiffusion(θ₀...)

Random.seed!(4)
function simulateSegment(::S) where S
    freq = 50000
    x0 = SVector(0.0)
    dt = 1/freq
    T = 8.0
    tt = 0.0:dt:T
    Wnr = Wiener{S}()
    WW = Bridge.samplepath(tt, zero(S))
    sample!(WW, Wnr)
    XX = solve(Euler(), x0, WW, P˟)

    XX.yy[1:freq:end], XX.tt[1:freq:end]
end
obs, obsTime = simulateSegment(0.0)

P̃ = [SinDiffusionAux(θ₀..., t₀, u[1], T, v[1]) for (t₀, T, u, v) in
        zip(obsTime[1:end-1], obsTime[2:end], obs[1:end-1], obs[2:end])]
Ls = [L for _ in P̃]
Σs = [Σ for _ in P̃]
τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
numSteps=1*10^4
tKernel = RandomWalk([1.0, 1.0, 1.0, 1.0], [false, false, false, true])

priors = Priors((Normal(1.0, 10.0),)) # 1.0, 1000.0
logpdf(P::Normal, θ) = -0.5*log(2.0*π*P.σ^2) - 0.5*((θ[4]-P.μ)/P.σ)^2
biasedPriors = Priors((Normal(7.0, 5.0),))# 6.0, 1.0
ladderOfPriors = LadderOfPriors((Priors((Normal(1.0, 10.0),)),
                                 Priors((Normal(7.0, 5.0),)),#7,5
                                 #Priors((Normal(7.0, 3.0),)),
                                 #Priors((Normal(7.0, 2.0),)),
                                 #Priors((Normal(7.0, 1.5),)),
                                 #Priors((Normal(7.0, 1.2),)),
                                 #Priors((Normal(7.0, 1.0),)),
                                ))
cs = [1.0, 2.0]#, 5.0, 8.0*10^1, 2.0*10^3, 1.0*10^5]#, 1.7*10^6]
fpt = [NaN for _ in P̃]

Random.seed!(4)
(chain, accRateImp, accRateUpdt,
    paths, time_) = mcmc(fptOrPartObs, obs, obsTime, x0, 0.0, P˟, P̃, Ls, Σs,
                         numSteps, tKernel, biasedPriors, τ;
                         fpt=fpt,
                         ρ=0.9,
                         dt=1/5000,
                         saveIter=3*10^2,
                         verbIter=10^2,
                         updtCoord=(#Val((true, false, false, false)),
                                    #Val((false, true, false, false)),
                                    #Val((false, false, true, false)),
                                    Val((false, false, false, true)),
                                    ),
                         paramUpdt=true,
                         updtType=(#MetropolisHastingsUpdt(),
                                   #MetropolisHastingsUpdt(),
                                   #MetropolisHastingsUpdt(),
                                   MetropolisHastingsUpdt(),
                                   ),
                         skipForSave=10^1,
                         solver=Vern7())

print("imputation acceptance rate: ", accRateImp,
      ", parameter update acceptance rate: ", accRateUpdt)

f1(x) = x
f2(x) = sin(x)
f3(x) = x^2

σchain = [θ[4] for θ in chain]
testMean = [mean(f.(σchain)) for f in [f1, f2, f3]]
ωs = exp.(logpdf.(priors[1], chain)-logpdf.(biasedPriors[1], chain))
testWeightedMean = [sum(f.(σchain) .* ωs)/sum(ωs) for f in [f1, f2, f3]]




print("Hello")










#thinnedChain = chain[1:length(priors)*3*10^2:end][2:end]

# save the results
#if parametrisation in (:simpleAlter, :complexAlter)
#    pathsToSave = [[alterToRegular(e, θ[1], θ[2]) for e in path] for (path,θ)
#                                      in zip(paths, thinnedChain)]
#    # only one out of many starting points will be plotted
#    x0 = alterToRegular(x0, chain[1][1], chain[1][2])
#elseif parametrisation in (:simpleConjug, :complexConjug)
#    pathsToSave = [[conjugToRegular(e, θ[1], 0) for e in path] for (path,θ)
#                                      in zip(paths, thinnedChain)]
#    x0 = conjugToRegular(x0, chain[1][1], 0)
#end
#
#df2 = savePathsToFile(pathsToSave, time_, joinpath(outdir, "sampled_paths.csv"))
#df3 = saveChainToFile(chain, joinpath(outdir, "chain.csv"))
#
#include("src/plots.jl")
## make some plots
#set_default_plot_size(30cm, 20cm)
#if fptObsFlag
#    plotPaths(df2, obs=[Float64.(df.upCross), [x0[2]]],
#              obsTime=[Float64.(df.time), [0.0]], obsCoords=[1,2])
#else
#    plotPaths(df2, obs=[Float64.(df.x1), [x0[2]]],
#              obsTime=[Float64.(df.time), [0.0]], obsCoords=[1,2])
#end
#plotChain(df3, coords=[1])
#plotChain(df3, coords=[2])
#plotChain(df3, coords=[3])
#plotChain(df3, coords=[5])
