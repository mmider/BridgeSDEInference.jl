using Bridge, StaticArrays, Distributions
using Statistics, Random, LinearAlgebra
using DataFrames
using CSV


SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

# choose parametrisation of the FitzHugh-Nagumo
POSSIBLE_PARAMS = [:regular, :simpleAlter, :complexAlter, :simpleConjug,
                   :complexConjug]
parametrisation = POSSIBLE_PARAMS[5]
include(joinpath(SRC_DIR, "fitzHughNagumo.jl"))
include(joinpath(SRC_DIR, "fitzHughNagumo_conjugateUpdt.jl"))

include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "vern7.jl"))
#include(joinpath(SRC_DIR, "tsit5.jl"))
#include(joinpath(SRC_DIR, "rk4.jl"))
include(joinpath(SRC_DIR, "ralston3.jl"))
include(joinpath(SRC_DIR, "priors.jl"))
include(joinpath(SRC_DIR, "guid_prop_bridge.jl"))
include(joinpath(SRC_DIR, "random_walk.jl"))
include(joinpath(SRC_DIR, "blocking_schedule.jl"))
include(joinpath(SRC_DIR, "starting_pt.jl"))
include(joinpath(SRC_DIR, "mcmc.jl"))
include(joinpath(SRC_DIR, "path_to_wiener.jl"))

include(joinpath(AUX_DIR, "read_and_write_data.jl"))
include(joinpath(AUX_DIR, "transforms.jl"))

# decide if first passage time observations or partially observed diffusion
fptObsFlag = true

# pick dataset
filename = "path_fpt_simpleConjug.csv"

# fetch the data
(df, x0, obs, obsTime, fpt,
      fptOrPartObs) = readData(Val(fptObsFlag), joinpath(OUT_DIR, filename))

# Initial parameter guess.
θ₀ = [10.0, -8.0, 25.0, 0.0, 3.0]
# Target law
P˟ = FitzhughDiffusion(θ₀...)
# Auxiliary law
P̃ = [FitzhughDiffusionAux(θ₀..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
     in zip(obsTime[1:end-1], obsTime[2:end], obs[1:end-1], obs[2:end])]

L = @SMatrix [1. 0.]
Σdiagel = 10^(-10)
Σ = @SMatrix [Σdiagel]

Ls = [L for _ in P̃]
Σs = [Σ for _ in P̃]
τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
numSteps=3*10^3
saveIter=3*10^2
tKernel = RandomWalk([3.0, 5.0, 0.5, 0.01, 0.5],
                     [false, false, false, false, true])
priors = Priors((#MvNormal([0.0,0.0,0.0], diagm(0=>[1000.0, 1000.0, 1000.0])),
                 #MvNormal([0.0], diagm(0=>[1000.0])),
                 ImproperPrior(),
                 #ImproperPrior(),)
                 ))
𝔅 = ChequeredBlocking()
blockingParams = (collect(1:length(obs)-2)[1:1:end], 10^(-7), SimpleChangePt(100))
changePt = NoChangePt()
x0Pr = KnownStartingPt(x0)
#x0Pr = GsnStartingPt(x0, x0, @SMatrix [20. 0; 0 20.])
warmUp=100

Random.seed!(4)
start = time()
(chain, accRateImp, accRateUpdt,
    paths, time_) = mcmc(eltype(x0), fptOrPartObs, obs, obsTime, x0Pr, 0.0, P˟,
                         P̃, Ls, Σs, numSteps, tKernel, priors, τ;
                         fpt=fpt,
                         ρ=0.95,
                         dt=1/500,
                         saveIter=saveIter,
                         verbIter=10^2,
                         updtCoord=(#Val((true, true, true, false, false)),
                                    #Val((true, false, false, false, false)),
                                    Val((false, false, true, false, false)),
                                    ),
                         paramUpdt=true,
                         updtType=(#ConjugateUpdt(),
                                   MetropolisHastingsUpdt(),
                                   #MetropolisHastingsUpdt(),
                                   ),
                         skipForSave=10^1,
                         blocking=𝔅,
                         blockingParams=blockingParams,
                         solver=Vern7(),
                         changePt=changePt,
                         warmUp=warmUp)
elapsed = time() - start
print("time elapsed: ", elapsed, "\n")

print("imputation acceptance rate: ", accRateImp,
      ", parameter update acceptance rate: ", accRateUpdt)

x0⁺, pathsToSave = transformMCMCOutput(x0, paths, saveIter; chain=chain, #chain=chain #θ=θ₀
                                       numGibbsSteps=1,
                                       parametrisation=parametrisation,
                                       warmUp=warmUp)

df2 = savePathsToFile(pathsToSave, time_, joinpath(OUT_DIR, "sampled_paths.csv"))
df3 = saveChainToFile(chain, joinpath(OUT_DIR, "chain.csv"))

include(joinpath(AUX_DIR, "plotting_fns.jl"))
set_default_plot_size(30cm, 20cm)
plotPaths(df2, obs=[Float64.(df.upCross), [x0⁺[2]]],
          obsTime=[Float64.(df.time), [0.0]], obsCoords=[1,2])

plotChain(df3, coords=[1])
plotChain(df3, coords=[2])
plotChain(df3, coords=[3])
plotChain(df3, coords=[5])
