# using DelimitedFiles
# using Makie
# data = readdlm("../LinneasData190920.csv", ';')
#
# data[isnan.(data)] .= circshift(data, (-1,0))[isnan.(data)]
# data[isnan.(data)] .= circshift(data, (1,0))[isnan.(data)]
# data[isnan.(data)] .= circshift(data, (-2,0))[isnan.(data)]
# data[isnan.(data)] .= circshift(data, (2,0))[isnan.(data)]
# data[isnan.(data)] .= circshift(data, (3,0))[isnan.(data)]
# any(isnan.(data))
#
# #data = replace(data, NaN=>missing)
# #μ = mapslices(mean∘skipmissing, data, dims=1)
# #sigma = mapslices(std∘skipmissing, data, dims=1)
# #surface(0..1, 0..5, data)



"""
    mcmc(ObsScheme::AbstractObsScheme, obs, obsTimes, yPr::StartingPtPrior, w, P˟, P̃, Ls, Σs,
         numSteps, tKernel, priors; fpt=fill(NaN, length(obsTimes)-1), ρ=0.0,
         dt=1/5000, timeChange=true, saveIter=NaN, verbIter=NaN,
         updtCoord=(Val((true,)),), paramUpdt=true, skipForSave=1,
         updtType=(MetropolisHastingsUpdt(),), solver::ST=Ralston3(), warmUp=0)

Gibbs sampler alternately imputing unobserved parts of the path and updating
unknown coordinates of the parameter vector (the latter only if paramUpdt==true)
...
# Arguments
- `ObsScheme`: observation scheme---first-passage time or partial observations
- `obs`: vector with observations
- `obsTimes`: times of the observations
- `yPr`: prior over the starting point of the diffusion path
- `w`: dummy variable whose type must agree with the type of the Wiener process
- `P˟`: law of the target diffusion (with initial θ₀ set)
- `P̃`: law of the auxiliary process (with initial θ₀ set)
- `Ls`: vector of observation operators (one per each observation)
- `Σs`: vector of covariance matrices of the noise (one per each observaiton)
- `numSteps`: number of mcmc iterations
- `tKernel`: transition kernel (also with initial θ₀ set)
- `priors`: a list of lists of priors
- `τ`: time-change transformation
- `fpt`: info about first-passage time conditioning
- `ρ`: memory parameter for the Crank-Nicolson scheme
- `dt`: time-distance for the path imputation
- `saveIter`: save path `XX` once every `saveIter` many iterations
- `verbIter`: print out progress info once every `verbIter` many iterations
- `updtCoord`: list of objects declaring indices of to-be-updated parameters
- `paramUpdt`: flag for whether to update parameters at all
- `skipForSave`: when saving paths, save only one in every `skipForSave` points
- `updtType`: list of types of updates to cycle through
- `solver`: numerical solver used for computing backward ODEs
- `warmUp`: number of steps for which no parameter update is to be made
...
"""
function mcmc(::Type{𝕂}, ObsScheme::AbstractObsScheme, obs, obsTimes, yPr::Vector{<:StartingPtPrior}, w,
              P˟, P̃, Ls, Σs, numSteps, tKernel, priors, τ;
              fpt=fill(NaN, size(obs)), # not sure if right size
              ρ=0.0, dt=1/5000, saveIter=NaN,
              verbIter=NaN, updtCoord=(Val((true,)),),
              paramUpdt=true,
              skipForSave=1, updtType=(MetropolisHastingsUpdt(),),
              blocking=NoBlocking(),
              blockingParams=([], 0.1, NoChangePt()),
              solver=Ralston3(), changePt::CP=NoChangePt(), warmUp=0
              ) where {𝕂, CP}

    K = length(obs)
    P = [findProposalLaw(𝕂, obs[k], obsTimes[k], P˟, P̃[k], Ls[k], Σs[k], τ; dt=dt, solver=solver,
                     changePt=CP(getChangePt(blockingParams[3])) ) for k in 1:K]
    N = length(obs[1])
    m = N - 1
    updtLen = length(updtCoord)
    tu = initialise(ObsScheme, P[1], m, yPr[1], w, fpt[1])
    vars = (:Wnr, :WWᵒ, :WW, :XXᵒ, :XX, :Pᵒ, :ll)
    local Wnr, WWᵒ, WW, XXᵒ, XX, Pᵒ, ll
    Wnr = [tu[1]]; WWᵒ = [tu[2]]; WW = [tu[3]]; XXᵒ= [tu[4]]; XX = [tu[5]]; Pᵒ = [tu[6]]; ll = [tu[7]]

    yPr[1] = tu[8]
    for k in 2:K
        tu = initialise(ObsScheme, P[k], m, yPr[k], w, fpt[k])
        for (i, var) in enumerate(vars)
            eval(:(push!($var, $tu[$i])))
        end
        yPr[k] = tu[end]
    end

    Paths = []
    accImpCounter = 0
    accUpdtCounter = [0 for i in 1:updtLen]
    θ = params(P˟)
    θchain = Vector{typeof(θ)}(undef, (numSteps-warmUp)*updtLen+1)
    θchain[1] = copy(θ)
    recomputeODEs = [any([e in dependsOnParams(P[1][1].Pt) for e
                         in idx(uc)]) for uc in updtCoord]

    updtStepCounter = 1
    𝔅 = [setBlocking(blocking, blockingParams, P[k], WW[k], XX[k]) for k in 1:K]
    #display(𝔅)

    for i in 1:numSteps
        verbose = (i % verbIter == 0)
        i > warmUp && savePath!(Paths, blocking == NoBlocking() ? XX : 𝔅.XX,
                                (i % saveIter == 0), skipForSave)
        for k in 1:K

            ll[k], acc[k], 𝔅[k], yPr[k] = impute!(ObsScheme[k], 𝔅[k], Wnr[k], yPr[k], WWᵒ[k], WW[k], XXᵒ[k], XX[k],
                                  P[k], ll[k], fpt[k], ρ=ρ, verbose=verbose, it=i,
                                  solver=solver)
        end
        accImpCounter += 1*acc
        if paramUpdt && i > warmUp
            for j in 1:updtLen
                (ll, acc, θ,
                 yPr) = updateParam!(ObsScheme, updtType[j], 𝔅, tKernel, θ,
                                     updtCoord[j], yPr, WW, Pᵒ, P, XXᵒ, XX, ll,
                                     priors[j], fpt, recomputeODEs[j];
                                     solver=solver, verbose=verbose, it=i)
                accUpdtCounter[j] += 1*acc
                updtStepCounter += 1
                θchain[updtStepCounter] = copy(θ)
                verbose && print("\n")
            end
            verbose && print("------------------------------------------------",
                             "------\n")
        end
    end
    displayAcceptanceRate(𝔅)
    Time = collect(Iterators.flatten(p.tt[1:skipForSave:end-1] for p in P))
    θchain, accImpCounter/numSteps, accUpdtCounter./numSteps, Paths, Time
end
