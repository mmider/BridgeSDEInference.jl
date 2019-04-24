mkpath("output/")
outdir="output"

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using ForwardDiff: value
const ℝ = SVector{N,T} where {N,T}
# specify observation scheme
L = @SMatrix [1.0]
Σdiagel = 10^(-5)
Σ = @SMatrix [Σdiagel]

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

logpdf(P::Normal, θ) = -0.5*log(2.0*π*P.σ^2) - 0.5*((θ[4]-P.μ)/P.σ)^2
priors = Priors((Normal(1.0, 10.0),))
fpt = [NaN for _ in P̃]

Random.seed!(4)
(chain, accRateImp, accRateUpdt,
    paths, time_) = mcmc(eltype(x0), fptOrPartObs, obs, obsTime, x0, 0.0, P˟, P̃, Ls, Σs,
                         numSteps, tKernel, priors, τ;
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
                         updtType=(#ConjugateUpdt(),
                                   #MetropolisHastingsUpdt(),
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
# estimate under true prior
straightMean = [mean(f.(σchain)) for f in [f1, f2, f3]]

#re-running with biased priors
P˟ = SinDiffusion(θ₀...)
P̃ = [SinDiffusionAux(θ₀..., t₀, u[1], T, v[1]) for (t₀, T, u, v) in
        zip(obsTime[1:end-1], obsTime[2:end], obs[1:end-1], obs[2:end])]
biasedPriors = Priors((Normal(7.0, 5.0),))
Random.seed!(4)
(chainB, accRateImpB, accRateUpdtB,
    pathsB, time_B) = mcmc(eltype(x0), fptOrPartObs, obs, obsTime, x0, 0.0, P˟, P̃, Ls, Σs,
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
                         updtType=(#ConjugateUpdt(),
                                   #MetropolisHastingsUpdt(),
                                   #MetropolisHastingsUpdt(),
                                   #MetropolisHastingsUpdt(),
                                   MetropolisHastingsUpdt(),
                                   ),
                         skipForSave=10^1,
                         solver=Vern7())

σchainB = [θ[4] for θ in chainB]
# these are weights
ωs = exp.(logpdf.(priors[1], chainB)-logpdf.(biasedPriors[1], chainB))
weightedMean = [sum(f.(σchainB) .* ωs)/sum(ωs) for f in [f1, f2, f3]]
