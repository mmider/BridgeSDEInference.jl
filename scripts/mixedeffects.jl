SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR = joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
using BridgeSDEInference

const BSI = BridgeSDEInference
using DataFrames, DelimitedFiles, CSV
using Test
using Makie
using Bridge, BridgeSDEInference, StaticArrays, Distributions
using Statistics, Random, LinearAlgebra

#include(joinpath(AUX_DIR, "read_and_write_data.jl"))
include(joinpath(AUX_DIR, "transforms.jl"))
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

data = cumsum(rand(200,100), dims=1) # Mock data

N, K = size(data)

x0 = [𝕏(x, 0.0) for x in data[:, 1]]
obs = map(𝕏, data)
obsTime = hcat([range(0, 1, length=N) for k in 1:K]...)
fpt = fill(NaN, size(data)) # really needed?
fptOrPartObs = PartObs()


param = :complexConjug
# Initial parameter guess.
θ₀ = (10.0, -8.0, 15.0, 0.0, 3.0)
randomEffects = (false, false, false, false, true)

# Target law
P˟ = [FitzhughDiffusion(param, θ₀...) for i in 1:K]

P̃ = map(CartesianIndices((N-1, K))) do I
      i, j = I[1], I[2]
      t₀, T, u, v = obsTime[i], obsTime[i+1], obs[i, j], obs[i+1, j]
      FitzhughDiffusionAux(param, θ₀..., t₀, u[1], T, v[1])
end
display(P̃[1,1])
𝕂 = Float64
L = @SMatrix [1. 0.]
Σdiagel = 1e-10
Σ = @SMatrix [Σdiagel]

Ls = [L for _ in P̃]
Σs = [Σ for _ in P̃]
τ(t₀,T) = (x) -> t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
numSteps=1*10^5
saveIter=3*10^2
tKernel = RandomWalk([3.0, 5.0, 5.0, 0.01, 0.5],
                     [false, false, false, false, true])
priors = Priors((MvNormal([0.0,0.0,0.0], diagm(0=>[1000.0, 1000.0, 1000.0])),
                 #ImproperPrior(),
                 ImproperPrior()))
𝔅 = NoBlocking()
blockingParams = ([], 0.1, NoChangePt())
changePt = NoChangePt()
#x0Pr = KnownStartingPt(x0)
x0Pr = [GsnStartingPt(x, x, @SMatrix [20. 0; 0 20.]) for x in x0]
warmUp = 100

Random.seed!(4)
start = time()
(chain, accRateImp, accRateUpdt,
    paths, time_) = BSI.mixedmcmc(𝕂, fptOrPartObs, obs, obsTime, x0Pr, 0.0, P˟,
                         P̃, Ls, Σs, numSteps, tKernel, priors, τ;
                         fpt=fpt,
                         ρ=0.975,
                         dt=1/1000,
                         saveIter=saveIter,
                         verbIter=10^2,
                         updtCoord=(Val((true, true, true, false, false)),
                                    #Val((true, false, false, false, false)),
                                    Val((false, false, false, false, true)),
                                    ),
                         randomEffects=randomEffects,
                         paramUpdt=true,
                         updtType=(ConjugateUpdt(),
                                   #MetropolisHastingsUpdt(),
                                   MixedEffectsMHUpdt(),
                                   ),
                         skipForSave=10^0,
                         blocking=𝔅,
                         blockingParams=blockingParams,
                         solver=Vern7(),
                         changePt=changePt,
                         warmUp=warmUp)
elapsed = time() - start
print("time elapsed: ", elapsed, "\n")

print("imputation acceptance rate: ", accRateImp,
      ", parameter update acceptance rate: ", accRateUpdt)

x0⁺, pathsToSave = transformMCMCOutput(x0, paths, saveIter; chain=chain,
                                       numGibbsSteps=2,
                                       parametrisation=param,
                                       warmUp=warmUp)


df2 = savePathsToFile(pathsToSave, time_, joinpath(OUT_DIR, "sampled_paths.csv"))
df3 = saveChainToFile(chain, joinpath(OUT_DIR, "chain.csv"))

include(joinpath(AUX_DIR, "plotting_fns.jl"))
set_default_plot_size(30cm, 20cm)
plotPaths(df2, obs=[Float64.(df.x1), [x0⁺[2]]],
          obsTime=[Float64.(df.time), [0.0]], obsCoords=[1,2])

plotChain(df3, coords=[1])
plotChain(df3, coords=[2])
plotChain(df3, coords=[3])
plotChain(df3, coords=[5])
