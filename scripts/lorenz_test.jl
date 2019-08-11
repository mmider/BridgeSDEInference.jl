SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
#include(joinpath(SRC_DIR, "fitzHughNagumo.jl"))
#include(joinpath(SRC_DIR, "fitzHughNagumo_conjugateUpdt.jl"))

include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "vern7.jl"))
include(joinpath(SRC_DIR, "tsit5.jl"))
include(joinpath(SRC_DIR, "rk4.jl"))
include(joinpath(SRC_DIR, "ralston3.jl"))
include(joinpath(SRC_DIR, "priors.jl"))
include(joinpath(SRC_DIR, "guid_prop_bridge.jl"))

include(joinpath(SRC_DIR, "bounded_diffusion_domain.jl"))
include(joinpath(SRC_DIR, "euler_maruyama_dom_restr.jl"))
include(joinpath(SRC_DIR, "lorenz_system.jl"))
include(joinpath(SRC_DIR, "lorenz_system_const_vola.jl"))

include(joinpath(SRC_DIR, "random_walk.jl"))
include(joinpath(SRC_DIR, "blocking_schedule.jl"))
include(joinpath(SRC_DIR, "starting_pt.jl"))
include(joinpath(SRC_DIR, "adaptation.jl"))
include(joinpath(SRC_DIR, "mcmc.jl"))
include(joinpath(SRC_DIR, "path_to_wiener.jl"))

using StaticArrays
using Distributions # to define priors
using Random        # to seed the random number generator
# Let's generate the data
# -----------------------
using Bridge
include(joinpath(AUX_DIR, "data_simulation_fns.jl"))
Random.seed!(4)
#θ₀ = [10.0, 28.0, 8.0/3.0, 3.0, 3.0, 3.0]
θˣ = [10.0, 28.0, 8.0/3.0, 3.0]
Pˣ = LorenzCV(θˣ...)

x0, dt, T = ℝ{3}(1.5, -1.5, 25.0), 1/5000, 4.0
tt = 0.0:dt:T
XX, _ = simulateSegment(ℝ{3}(0.0, 0.0, 0.0), x0, Pˣ, tt)

θ₀ = θˣ
#θ₀ = [5.0, 15.0, 6.0, 8.0]
#Pˣ = LorenzCV(θ₀...)


skip = 200

Σdiagel = 10^0
Σ = SMatrix{2,2}(1.0I)*Σdiagel
L = @SMatrix[1.0 0.0 0.0;
             0.0 1.0 0.0]



obsTime, obsVals = XX.tt[1:skip:end], [rand(Gaussian(L*x, Σ)) for x in XX.yy[1:skip:end]]
obsVals
fptOrPartObs = PartObs()
fpt = [NaN for _ in obsTime[2:end]]

auxFlag = Val{(true,true,false)}()
P̃ = [LorenzCVAux(θ₀..., t₀, u, T, v, auxFlag, x0[3]) for (t₀, T, u, v)
     in zip(obsTime[1:end-1], obsTime[2:end], obsVals[1:end-1], obsVals[2:end])]


Ls = [L for _ in P̃]
Σs = [Σ for _ in P̃]
τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
numSteps=2*10^3
saveIter=1*10^2


tKernel = RandomWalk([2.0, 3.0, 0.64, 0.8], #[2.0, 1.0, 0.64, 0.3]
                     [false, false, false, true])

priors = Priors((ImproperPrior(), ImproperPrior(), ImproperPrior(),
                 ImproperPrior()))

𝔅 = NoBlocking()
blockingParams = ([], 0.1, NoChangePt())
changePt = NoChangePt()
x0Pr = KnownStartingPt(x0)
#x0Pr = GsnStartingPt(x0, x0, @SMatrix [20.0 0.0 0.0; 0.0 20.0 0.0; 0.0 0.0 400.0])
warmUp = 100

#adaptation = NoAdaptation()
adaptation = Adaptation(ℝ{3}(0.0, 0.0, 0.0), [0.85, 0.7, 0.6], [0.5, 0.2, 0.0], [500, 500, 500], 1)

Random.seed!(4)
start = time()
(chain, accRateImp, accRateUpdt,
    paths, time_) = mcmc(eltype(x0), fptOrPartObs, obsVals, obsTime, x0Pr,
                         ℝ{3}(0.0, 0.0, 0.0), Pˣ, P̃, Ls, Σs, numSteps,
                         tKernel, priors, τ;
                         fpt=fpt,
                         ρ=0.9,
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
                         adaptiveProp=adaptation)
elapsed = time() - start
print("time elapsed: ", elapsed, "\n")


print("imputation acceptance rate: ", accRateImp,
      ", parameter update acceptance rate: ", accRateUpdt)

using Plots
pTp = [[[x[i] for x in path] for path in paths] for i in 1:3]

function plotPaths(j, obsIdxS, obsIdxE, show_obs=true, half=1)
    idxS = div((obsIdxS-1)*skip,2)+1
    idxE = div((obsIdxE-1)*skip,2)+1
    print(idxS, ", ", idxE, "\n")
    p = plot()
    iRange = (half == 1) ? (1:div(length(paths),2)) : (div(length(paths),2):length(paths))

    for i in iRange
        plot!(time_[idxS:idxE], pTp[j][i][idxS:idxE], label="", color="steelblue", alpha=0.2, linewidth=0.2)
    end
    if show_obs
        scatter!(obsTime[obsIdxS:obsIdxE], [x[j] for x in obsVals][obsIdxS:obsIdxE],
                 color="orange", label="")
    end
    p
end

plotPaths(1, 1, 10)
plotPaths(1, 1, 10, true, 2)
plotPaths(2, 1, 10)
plotPaths(2, 1, 10, true, 2)
plotPaths(3, 1, 10, false)
plotPaths(3, 1, 10, false, 2)




plot([θ[1] for θ in chain])
plot([θ[2] for θ in chain])
plot([θ[3] for θ in chain])
plot([θ[4] for θ in chain])

#=

ws.P

updateLaws!(ws.Pᵒ, params(ws.P[1].Target))
solveBackRec!(NoBlocking(), ws.Pᵒ, Vern7())
ws.P[1].Pt.X̄(0.05)
ws.Pᵒ[1].Pt.X̄(0.05)
m = length(ws.P)
pathLogLikhd(PartObs(), ws.XX, ws.P, 1:m, ws.fpt)
pathLogLikhd(PartObs(), ws.XX, ws.Pᵒ, 1:m, ws.fpt)



s
function plot_ll(xx, i, θᵒ)
    llᵒpath = zeros(Float64, length(xx))
    llᵒobs = zeros(Float64, length(xx))
    llᵒtotal = zeros(Float64, length(xx))
    m = length(ws.Pᵒ)
    y = ws.XX[1].yy[1]
    for (j,x) in enumerate(xx)
        θₓ = copy(θᵒ)
        θₓ[i] = x
        updateLaws!(ws.Pᵒ, θₓ)
        solveBackRec!(NoBlocking(), ws.Pᵒ, Vern7())
        findPathFromWiener!(ws.XXᵒ, y, ws.WW, ws.Pᵒ, 1:m)
        llᵒpath[j] = pathLogLikhd(PartObs(), ws.XXᵒ, ws.Pᵒ, 1:m, ws.fpt)
        llᵒobs[j] = lobslikelihood(ws.Pᵒ[1], y)
        llᵒtotal[j] = llᵒpath[j] + llᵒobs[j]
    end
    p1 = plot(xx, llᵒpath, label="path loglikhd")
    solveBackRec!(NoBlocking(), ws.P, Vern7())
    llpath = pathLogLikhd(PartObs(), ws.XX, ws.P, 1:m, ws.fpt)
    llobs = lobslikelihood(ws.P[1], y)
    scatter!([θᵒ[i]], [llpath])


    p2 = plot(xx, llᵒobs, label="obs loglikhd")
    plot!(xx, llᵒtotal, label="total loglikhd")
    scatter!([θᵒ[i]], [llobs])
    scatter!([θᵒ[i]], [llpath + llobs])
    p = plot(p1,p2,layout=(1,2),legend=false)
    p
end

p = plot_ll(9.0:0.01:12.0, 1, params(ws.P[1].Target))

=#
