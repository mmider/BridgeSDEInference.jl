mkpath("output/")
outdir="output"

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models: ℝ
using DataFrames
using CSV

L = @SMatrix [1.0]
Σ = @SMatrix[10^(-6)]

include("src/double_well_potential.jl")
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

include("src/ladders.jl")
include("src/temperature_mcmc.jl")

x0 = ℝ{1}(0.0)

fptOrPartObs = PartObs()
θ₀ = [1.0, 2.0, 0.5]
P˟ = DoubleWellPotential(θ₀...)

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
#obs
#obs = ℝ{1}.([0.0, 0.0])
#obsTime = [0.0, 8.0]

P̃ = [DoubleWellPotentialAux(θ₀..., t₀, u[1], T, v[1]) for (t₀, T, u, v) in
        zip(obsTime[1:end-1], obsTime[2:end], obs[1:end-1], obs[2:end])]
Ls = [L for _ in P̃]
Σs = [Σ for _ in P̃]
τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
numSteps=1*10^4
tKernel = RandomWalk([1.0, 1.0, 1.0], [false, false, true])

priors = Priors((Normal(1.0, 10.0),)) # 1.0, 1000.0
logpdf(P::Normal, θ) = -0.5*log(2.0*π*P.σ^2) - 0.5*((θ[3]-P.μ)/P.σ)^2
biasedPriors = Priors((Normal(7.0, 5.0),))# 6.0, 1.0
ladderOfPriors = LadderOfPriors((Priors((Normal(1.0, 10.0),)),
                                 Priors((Normal(7.0, 5.0),)),#7,5
                                 Priors((Normal(7.0, 3.0),)),
                                 Priors((Normal(7.0, 2.0),)),
                                 Priors((Normal(7.0, 1.5),)),
                                 Priors((Normal(7.0, 1.2),)),
                                 #Priors((Normal(7.0, 1.0),)),
                                ))
cs = [1.0, 2.0, 5.0, 8.0*10^1, 2.0*10^3, 1.0*10^5]#, 1.7*10^6]
fpt = [NaN for _ in P̃]

mcmcParams = MCMCParams(obs=obs, obsTimes=obsTime, priors=priors, fpt=fpt,
                        ρ=0.9, dt=1/5000, saveIter=3*10^2, verbIter=10^2,
                        updtCoord=(#Val((true, false, false, false)),
                                   #Val((false, true, false, false)),
                                   #Val((false, false, true, false)),
                                   Val((false, false, true)),
                                   ),
                        paramUpdt=true, skipForSave=10^1,
                        updtType=(#MetropolisHastingsUpdt(),
                                  #MetropolisHastingsUpdt(),
                                  #MetropolisHastingsUpdt(),
                                  MetropolisHastingsUpdt(),
                                  ),
                        cs=cs,
                        biasedPriors=biasedPriors,
                        ladderOfPriors=ladderOfPriors,
                        𝓣Ladder=NaN
                        )

Random.seed!(4)
(chain, 𝓣chain, logωs, accRateImp, accRateUpdt, accptRate𝓣, paths, 𝓣chainPth,
    time_) = wmcmc(VanillaMCMC(), fptOrPartObs, x0, 0.0, P˟, P̃, Ls, Σs,
                   numSteps, tKernel, τ, mcmcParams; solver=Vern7())

print("imputation acceptance rate: ", accRateImp,
      ", parameter update acceptance rate: ", accRateUpdt,
      ", temperature acceptance rate: "); display(accptRate𝓣)

df2 = savePathsToFile(paths, time_, joinpath(outdir, "sampled_paths.csv"))
df3 = saveChainToFile(chain, joinpath(outdir, "chain.csv"))

include("src/plots.jl")
# make some plots
set_default_plot_size(30cm, 20cm)

plot(df2, x=:time, y=:x1, color=:idx, Geom.line,
     Scale.color_continuous(colormap=Scale.lab_gradient("#fceabb", "#a2acae",
                                                        "#36729e")))
ιchain = [ι for (ι,_,_) in  𝓣chain]


plot(df3, y=:x1, Geom.line)
plot(df3, y=:x1_1, Geom.line)
plot(df3, y=:x1_2, Geom.line)
plot(df3, y=:x1_3, Geom.line)
plot(y=ιchain, Geom.line)
θchain = [θ[3] for θ in chain]
θchain1 = [θ[3] for (θ,(ι,_,_)) in  zip(chain, 𝓣chain) if ι == 1]
θchain2 = [θ[3] for (θ,(ι,_,_)) in  zip(chain, 𝓣chain) if ι == 2]
plot(y=θchain1, Geom.line)
plot(y=θchain2, Geom.line)

f1(x) = x
f2(x) = sin(x)
f3(x) = x^2



ωs = exp.(logωs)
plot(x=ωs, Geom.histogram)
testsM = [mean(f.(df3.x1_3) .* ωs) for f in [f1, f2, f3]]
testsW = [sum(f.(df3.x1_3) .* ωs)/sum(ωs) for f in [f1, f2, f3]]
for i in 1:10
    js = rand(1:length(ωs), 1000)
    print([sum(f.(df3.x1_3[js]) .* ωs[js])/sum(ωs[js]) for f in [f1, f2, f3]], "\n")
end

tests = [mean(f.(df3.x1_3)) for f in [f1, f2, f3]]

for i in 1:10
    js = rand(1:length(θchain1), 100)
    teston1 = [mean(f.(θchain1[js])) for f in [f1, f2, f3]]
    print(teston1, "\n")
end

teston1 = [mean(f.(θchain1)) for f in [f1, f2, f3]]
