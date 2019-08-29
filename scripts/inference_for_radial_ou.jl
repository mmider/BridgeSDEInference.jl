# ------------------------------------------------------------
# NOTE BROKEN AND POSSIBLY BEYOND REPAIR                     |
# explosion of likelihoods seems to make inference impossible|
# ------------------------------------------------------------

SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
include(joinpath(SRC_DIR, "fitzHughNagumo.jl"))

include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "vern7.jl"))
include(joinpath(SRC_DIR, "tsit5.jl"))
include(joinpath(SRC_DIR, "rk4.jl"))
include(joinpath(SRC_DIR, "ralston3.jl"))
include(joinpath(SRC_DIR, "priors.jl"))

include(joinpath(SRC_DIR, "bounded_diffusion_domain.jl"))
include(joinpath(SRC_DIR, "radial_ornstein_uhlenbeck.jl"))
include(joinpath(SRC_DIR, "euler_maruyama_dom_restr.jl"))

include(joinpath(SRC_DIR, "guid_prop_bridge.jl"))
include(joinpath(SRC_DIR, "conjugateUpdt.jl"))
include(joinpath(SRC_DIR, "random_walk.jl"))
include(joinpath(SRC_DIR, "blocking_schedule.jl"))
include(joinpath(SRC_DIR, "starting_pt.jl"))
include(joinpath(SRC_DIR, "mcmc.jl"))
include(joinpath(SRC_DIR, "path_to_wiener.jl"))



using StaticArrays
using Distributions # to define priors
using Random        # to seed the random number generator




# Let's generate the data
# -----------------------
using Bridge
#import Main.BridgeSDEInference.forcedSolve
include(joinpath(AUX_DIR, "data_simulation_fns.jl"))
Random.seed!(4)
θ₀ = [0.05, √2.0]
Pˣ = RadialOU(θ₀...)

x0, dt, T = ℝ{1}(0.5), 1/5000, 1.0
tt = 0.0:dt:T
XX, _ = simulateSegment(0.0, x0, Pˣ, tt)

num_obs = 11
skip = div(length(tt), num_obs-1)
obsTime, obsVals = collect(tt)[1:skip:end], XX.yy[1:skip:end]

fptOrPartObs = PartObs()
fpt = [NaN for _ in obsTime[2:end]]

P̃ = [RadialOUAux(θ₀..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
     in zip(obsTime[1:end-1], obsTime[2:end], obsVals[1:end-1], obsVals[2:end])]

L = @SMatrix [1.]
Σdiagel = 10^(-3)
Σ = @SMatrix [Σdiagel]

Ls = [L for _ in P̃]
Σs = [Σ for _ in P̃]

τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
numSteps=1*10^1
saveIter=1*10^0

tKernel = RandomWalk([0.002, 0.1], [true, true])
priors = Priors((ImproperPrior(), ImproperPrior()))

𝔅 = NoBlocking()
blockingParams = ([], 0.1, NoChangePt())
changePt = NoChangePt()
x0Pr = KnownStartingPt(x0)
warmUp = 0

Random.seed!(4)
start = time()
(chain, accRateImp, accRateUpdt,
    paths, time_) = mcmc(eltype(x0), fptOrPartObs, obsVals, obsTime, x0Pr, 0.0,
                         Pˣ, P̃, Ls, Σs, numSteps, tKernel, priors, τ;
                         fpt=fpt,
                         ρ=0.975,
                         dt=1/1000,
                         saveIter=saveIter,
                         verbIter=10^2,
                         updtCoord=(Val((true, false)),
                                    Val((false, true)),
                                    ),
                         paramUpdt=false,
                         updtType=(MetropolisHastingsUpdt(),
                                   MetropolisHastingsUpdt(),
                                   ),
                         skipForSave=10^0,
                         blocking=𝔅,
                         blockingParams=blockingParams,
                         solver=Vern7(),
                         changePt=changePt,
                         warmUp=warmUp)
elapsed = time() - start
print("time elapsed: ", elapsed, "\n")

Xs = [[x[1] for x in path] for path in paths]

using Plots
p = plot(time_, Xs[1], color="steelblue", alpha=0.5, label="", ylims=[0,5])
scatter!(obsTime, [x[1] for x in obsVals])
display(p)
