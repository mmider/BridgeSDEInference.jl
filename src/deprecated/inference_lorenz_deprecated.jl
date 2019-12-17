SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
include(joinpath(SRC_DIR, "BridgeSDEInference_for_tests.jl"))


using StaticArrays
using Distributions # to define priors
using Random        # to seed the random number generator
# Let's generate the data
# -----------------------
using Bridge
include(joinpath(SRC_DIR, "auxiliary", "data_simulation_fns.jl"))
Random.seed!(4)
#θ₀ = [10.0, 28.0, 8.0/3.0, 3.0, 3.0, 3.0]
θˣ = [10.0, 28.0, 8.0/3.0, 3.0]
Pˣ = LorenzCV(θˣ...)

x0, dt, T = ℝ{3}(1.5, -1.5, 25.0), 1/5000, 10.0
tt = 0.0:dt:T
XX, _ = simulate_segment(ℝ{3}(0.0, 0.0, 0.0), x0, Pˣ, tt)


θ₀ = [5.0, 15.0, 6.0, 8.0]
Pˣ = LorenzCV(θ₀...)


skip = 1000

Σdiagel = 10^0
Σ = SMatrix{2,2}(1.0I)*Σdiagel
L = @SMatrix[1.0 0.0 0.0;
             0.0 1.0 0.0]



obsTime, obsVals = XX.tt[1:skip:end], [rand(Gaussian(L*x, Σ)) for x in XX.yy[1:skip:end]]

fptOrPartObs = PartObs()
fpt = [NaN for _ in obsTime[2:end]]

auxFlag = Val{(true,true,false)}()
P̃ = [LorenzCVAux(θ₀..., t₀, u, T, v, auxFlag, x0[3]) for (t₀, T, u, v)
     in zip(obsTime[1:end-1], obsTime[2:end], obsVals[1:end-1], obsVals[2:end])]


Ls = [L for _ in P̃]
Σs = [Σ for _ in P̃]
τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
numSteps=1*10^4
saveIter=1*10^3


tKernel = RandomWalk([2.0, 1.0, 0.64, 0.3],
                     [false, false, false, true])

priors = Priors((ImproperPrior(), ImproperPrior(), ImproperPrior(),
                 ImproperPrior()))

𝔅 = NoBlocking()
blockingParams = ([], 0.1, NoChangePt())
changePt = NoChangePt()
#x0Pr = KnownStartingPt(x0)
x0Pr = GsnStartingPt(x0, x0, @SMatrix [20.0 0.0 0.0; 0.0 20.0 0.0; 0.0 0.0 400.0])
warmUp = 100

Random.seed!(4)
start = time()
(chain, accRateImp, accRateUpdt,
    paths, time_) = mcmc(eltype(x0), fptOrPartObs, obsVals, obsTime, x0Pr,
                         ℝ{3}(0.0, 0.0, 0.0), Pˣ, P̃, Ls, Σs, numSteps,
                         tKernel, priors, τ;
                         fpt=fpt,
                         ρ=0.995,
                         dt=1/2000,
                         saveIter=saveIter,
                         verbIter=10^2,
                         updtCoord=(Val((true, false, false, false)),
                                    Val((false, true, false, false)),
                                    Val((false, false, true, false)),
                                    Val((false, false, false, true))
                                    ),
                         paramUpdt=true,
                         updtType=(MetropolisHastingsUpdt(),
                                   MetropolisHastingsUpdt(),
                                   MetropolisHastingsUpdt(),
                                   MetropolisHastingsUpdt()
                                   ),
                         skipForSave=10^0,
                         blocking=𝔅,
                         blockingParams=blockingParams,
                         solver=Vern7(),
                         changePt=changePt,
                         warmUp=warmUp,
                         adaptiveProp=NoAdaptation())
elapsed = time() - start
print("time elapsed: ", elapsed, "\n")


print("imputation acceptance rate: ", accRateImp,
      ", parameter update acceptance rate: ", accRateUpdt)

using Plots
pTp = [[[x[i] for x in path] for path in paths] for i in 1:3]

function plotPaths(j, obsIdxS, obsIdxE, show_obs=true)
    idxS = div((obsIdxS-1)*skip,5)+1
    idxE = div((obsIdxE-1)*skip,5)+1
    p = plot()
    for i in 1:length(paths)
        plot!(time_[idxS:idxE], pTp[j][i][idxS:idxE], label="", color="steelblue", alpha=0.2, linewidth=0.2)
    end
    if show_obs
        scatter!(obsTime[obsIdxS:obsIdxE], [x[j] for x in obsVals][obsIdxS:obsIdxE],
                 color="orange", label="")
    end
    p
end

plotPaths(1, 1, 10)
plotPaths(2, 1, 10)
plotPaths(3, 1, 10, false)

plot([θ[1] for θ in chain])
plot([θ[2] for θ in chain])
plot([θ[3] for θ in chain])
plot([θ[4] for θ in chain])
