#NOTE deprecating an substituting with Workspace
"""
    initialise(::ObsScheme, P, m, y::StartingPtPrior{T}, ::S, fpt)

Initialise the workspace for MCMC algorithm. Initialises containers for driving
Wiener processes `WWᵒ` & `WW`, for diffusion processes `XXᵒ` & `XX`, for
diffusion Law `Pᵒ` (parametetrised by proposal parameters) and defines the type
of Wiener process `Wnr`.
"""
function initialise(::ObsScheme, P, m, yPr::StartingPtPrior{T}, ::S,
                    fpt) where {ObsScheme <: AbstractObsScheme,T,S}
    y = startPt(yPr)
    Pᵒ = deepcopy(P)
    TW = typeof(sample([0], Wiener{S}()))
    TX = typeof(SamplePath([], zeros(T, 0)))
    XXᵒ = Vector{TX}(undef,m)
    WWᵒ = Vector{TW}(undef,m)
    Wnr = Wiener{S}()
    for i in 1:m
        WWᵒ[i] = Bridge.samplepath(P[i].tt, zero(S))
        sample!(WWᵒ[i], Wnr)
        WWᵒ[i], XXᵒ[i] = forcedSolve(Euler(), y, WWᵒ[i], P[i])    # this will enforce adherence to domain
        while !checkFpt(ObsScheme(), XXᵒ[i], fpt[i])
            sample!(WWᵒ[i], Wnr)
            forcedSolve!(Euler(), XXᵒ[i], y, WWᵒ[i], P[i])    # this will enforce adherence to domain
        end
        y = XXᵒ[i].yy[end]
    end
    y = startPt(yPr)
    ll = logpdf(yPr, y)
    ll += pathLogLikhd(ObsScheme(), XXᵒ, P, 1:m, fpt, skipFPT=true)
    ll += lobslikelihood(P[1], y)

    XX = deepcopy(XXᵒ)
    WW = deepcopy(WWᵒ)
    # needed for proper initialisation of the Crank-Nicolson scheme
    yPr = invStartPt(y, yPr, P[1])

    Wnr, WWᵒ, WW, XXᵒ, XX, Pᵒ, ll, yPr
end


#NOTE deprecated
"""
    swapXX!(𝔅::ChequeredBlocking, XX)

Swap containers between `XX` and `𝔅.XX`
"""
function swapXX!(𝔅::BlockingSchedule, XX)
    for block in 𝔅.blocks[𝔅.idx]
        swap!(XX, 𝔅.XX, block)
    end
end

#NOTE deprecated
"""
    swapXX!(𝔅::NoBlocking, XX)

nothing to do
"""
swapXX!(𝔅::NoBlocking, XX) = nothing



#NOTE deprecated
"""
    impute!(::ObsScheme, 𝔅::ChequeredBlocking, Wnr, y, WWᵒ, WW, XXᵒ, XX, P, ll,
            fpt; ρ=0.0, verbose=false, it=NaN, headStart=false) where
            ObsScheme <: AbstractObsScheme -> acceptedLogLikhd, acceptDecision

Imputation step of the MCMC scheme (without blocking).
...
# Arguments
- `::ObsScheme`: observation scheme---first-passage time or partial observations
- `𝔅`: object with relevant information about blocking
- `Wnr`: type of the Wiener process
- `yPr`: prior over the starting point of the diffusion path
- `WWᵒ`: containers for proposal Wiener paths
- `WW`: containers with old Wiener paths
- `XXᵒ`: containers for proposal diffusion paths
- `XX`: containers with old diffusion paths
- `P`: laws of the diffusion path (proposal and target)
- `11`: log-likelihood of the old (previously accepted) diffusion path
- `fpt`: info about first-passage time conditioning
- `ρ`: memory parameter for the Crank-Nicolson scheme
- `verbose`: whether to print updates info while sampling
- `it`: iteration index of the MCMC algorithm
- `headStart`: flag for whether to 'ease into' fpt conditions
...
"""
function impute!_deprecated(::ObsScheme, 𝔅::ChequeredBlocking, Wnr, yPr, WWᵒ, WW, XXᵒ, XX,
                 P, ll, fpt; ρ=0.0, verbose=false, it=NaN, headStart=false,
                 solver::ST=Ralston3()) where
                 {ObsScheme <: AbstractObsScheme, ST}
    θ = params(P[1].Target)             # current parameter
    𝔅 = next(𝔅, XX, θ)
    solveBackRec!(𝔅, 𝔅.P, ST())         # compute (H, Hν, c) for given blocks

    swapXX!(𝔅, XX)                      # move current path to object 𝔅
    noiseFromPath!(𝔅, 𝔅.XX, 𝔅.WW, 𝔅.P) # find noise WW that generates XX under 𝔅

    # compute white noise generating starting point under 𝔅
    yPr𝔅 = invStartPt(𝔅.XX[1].yy[1], yPr, 𝔅.P[1])

    for (blockIdx, block) in enumerate(𝔅.blocks[𝔅.idx])
        blockFlag = Val{block[1]}()
        y = 𝔅.XX[block[1]].yy[1]       # current starting point

        # set the starting point for the block
        yᵒ, yPrᵒ = proposalStartPt(𝔅, blockFlag, y, yPr𝔅, 𝔅.P[block[1]], ρ)

        # sample path in block
        sampleSegments!(block, Wnr, 𝔅.WW, 𝔅.WWᵒ, 𝔅.P, yᵒ, 𝔅.XXᵒ, ρ)
        setEndPtManually!(𝔅, blockIdx, block)

        # loglikelihoods
        llᵒ = startPtLogPdf(blockFlag, yPrᵒ, yᵒ)
        llᵒ += pathLogLikhd(ObsScheme(), 𝔅.XXᵒ, 𝔅.P, block, fpt)
        llᵒ += lobslikelihood(𝔅.P[block[1]], yᵒ)

        llPrev = startPtLogPdf(blockFlag, yPr𝔅, y)
        llPrev += pathLogLikhd(ObsScheme(), 𝔅.XX, 𝔅.P, block, fpt; skipFPT=true)
        llPrev += lobslikelihood(𝔅.P[block[1]], y)

        printInfo(verbose, it, value(llPrev), value(llᵒ), "impute")
        if acceptSample(llᵒ-llPrev, verbose)
            swap!(𝔅.XX, 𝔅.XXᵒ, block)
            registerAccpt!(𝔅, blockIdx, true)
            yPr𝔅 = yPrᵒ # can do something non-trivial only for the first block
        else
            registerAccpt!(𝔅, blockIdx, false)
        end
    end
    swapXX!(𝔅, XX) # move accepted path from object 𝔅 to general container XX
    noiseFromPath!(𝔅, XX, WW, P) # compute noise WW that generated XX under law P
    # compute white noise generating starting point under P
    y = XX[1].yy[1]
    yPr = invStartPt(y, yPr𝔅, P[1])

    ll = logpdf(yPr, y) # starting point contribution
    ll += pathLogLikhd(ObsScheme(), XX, P, 1:length(P), fpt; skipFPT=true)
    ll += lobslikelihood(P[1], y)

    # acceptance indicator does not matter for sampling with blocking
    return ll, true, 𝔅, yPr
end



struct Workspace{ObsScheme,S,TX,TW,R,Q}
    Wnr::Wiener{S}
    XXᵒ::Vector{TX}
    XX::Vector{TX}
    WWᵒ::Vector{TW}
    WW::Vector{TW}
    Pᵒ::Vector{R}
    P::Vector{R}
    fpt::Vector
    ρ::Float64 #TODO use vector instead for blocking
    result::Vector{Q}
    resultᵒ::Vector{Q}

    function Workspace(::ObsScheme, P::Vector{R}, m, yPr::StartingPtPrior{T},
                       ::S, fpt, ρ, updtCoord) where {ObsScheme <: AbstractObsScheme,R,T,S}
        y = startPt(yPr)
        Pᵒ = deepcopy(P)
        TW = typeof(sample([0], Wiener{S}()))
        TX = typeof(SamplePath([], zeros(T, 0)))
        XXᵒ = Vector{TX}(undef,m)
        WWᵒ = Vector{TW}(undef,m)
        Wnr = Wiener{S}()
        for i in 1:m
            WWᵒ[i] = Bridge.samplepath(P[i].tt, zero(S))
            sample!(WWᵒ[i], Wnr)
            WWᵒ[i], XXᵒ[i] = forcedSolve(Euler(), y, WWᵒ[i], P[i])    # this will enforce adherence to domain
            while !checkFpt(ObsScheme(), XXᵒ[i], fpt[i])
                sample!(WWᵒ[i], Wnr)
                forcedSolve!(Euler(), XXᵒ[i], y, WWᵒ[i], P[i])    # this will enforce adherence to domain
            end
            y = XXᵒ[i].yy[end]
        end
        y = startPt(yPr)
        ll = logpdf(yPr, y)
        ll += pathLogLikhd(ObsScheme(), XXᵒ, P, 1:m, fpt, skipFPT=true)
        ll += lobslikelihood(P[1], y)

        XX = deepcopy(XXᵒ)
        WW = deepcopy(WWᵒ)
        # needed for proper initialisation of the Crank-Nicolson scheme
        yPr = invStartPt(y, yPr, P[1])

        θ = params(P[1].Target)
        ϑs = [[θ[j] for j in idx(uc)] for uc in updtCoord]

        result = [DiffResults.GradientResult(ϑ) for ϑ in ϑs]
        resultᵒ = [DiffResults.GradientResult(ϑ) for ϑ in ϑs]
        Q = eltype(result)

        new{ObsScheme,S,TX,TW,R,Q}(Wnr, XXᵒ, XX, WWᵒ, WW, Pᵒ, P, fpt, ρ, result, resultᵒ), ll, yPr
    end

    function Workspace(𝓦𝓢::Workspace{ObsScheme,S,TX,TW,R,Q}, new_ρ::Float64
                       ) where {ObsScheme,S,TX,TW,R,Q}
        new{ObsScheme,S,TX,TW,R,Q}(𝓦𝓢.Wnr, 𝓦𝓢.XXᵒ, 𝓦𝓢.XX, 𝓦𝓢.WWᵒ, 𝓦𝓢.WW,
                                   𝓦𝓢.Pᵒ, 𝓦𝓢.P, 𝓦𝓢.fpt, new_ρ, 𝓦𝓢.result,
                                   𝓦𝓢.resultᵒ)
    end
end




"""
    mcmc(::ObsScheme, obs, obsTimes, yPr::StartingPtPrior, w, P˟, P̃, Ls, Σs,
         numSteps, tKernel, priors; fpt=fill(NaN, length(obsTimes)-1), ρ=0.0,
         dt=1/5000, timeChange=true, saveIter=NaN, verbIter=NaN,
         updtCoord=(Val((true,)),), paramUpdt=true, skipForSave=1,
         updtType=(MetropolisHastingsUpdt(),), solver::ST=Ralston3(), warmUp=0)

Gibbs sampler alternately imputing unobserved parts of the path and updating
unknown coordinates of the parameter vector (the latter only if paramUpdt==true)
...
# Arguments
- `::ObsScheme`: observation scheme---first-passage time or partial observations
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
function mcmc(::Type{K}, ::ObsScheme, obs, obsTimes, yPr::StartingPtPrior, w,
              P˟, P̃, Ls, Σs, numSteps, tKernel, priors, τ;
              fpt=fill(NaN, length(obsTimes)-1), ρ=0.0, dt=1/5000, saveIter=NaN,
              verbIter=NaN, updtCoord=(Val((true,)),), paramUpdt=true,
              skipForSave=1, updtType=(MetropolisHastingsUpdt(),),
              blocking::Blocking=NoBlocking(),
              blockingParams=([], 0.1, NoChangePt()),
              solver::ST=Ralston3(), changePt::CP=NoChangePt(), warmUp=0,
              adaptiveProp=NoAdaptation()
              ) where {K, ObsScheme <: AbstractObsScheme, ST, Blocking, CP}
    P = findProposalLaw( updtType[1], K, obs, obsTimes, P˟, P̃, Ls, Σs, τ; dt=dt, solver=ST(),
                         changePt=CP(getChangePt(blockingParams[3])) )
    m = length(obs)-1
    updtLen = length(updtCoord)
    𝓦𝓢, ll, yPr = Workspace(ObsScheme(), P, m, yPr, w, fpt, ρ, updtCoord)
    init_adaptation!(adaptiveProp, 𝓦𝓢)

    Paths = []
    accImpCounter = 0
    accUpdtCounter = [0 for i in 1:updtLen]
    θ = params(P˟)
    θchain = Vector{typeof(θ)}(undef, (numSteps-warmUp)*updtLen+1)
    θchain[1] = copy(θ)
    recomputeODEs = [any([e in dependsOnParams(P[1].Pt) for e
                         in idx(uc)]) for uc in updtCoord]

    updtStepCounter = 1
    𝔅 = setBlocking(blocking, blockingParams, 𝓦𝓢)
    display(𝔅)
    for i in 1:numSteps
        #print(i, ", ")
        verbose = (i % verbIter == 0)
        i > warmUp && savePath!(Paths, blocking == NoBlocking() ? 𝓦𝓢.XX : 𝔅.XX,
                                (i % saveIter == 0), skipForSave)
        ll, acc, 𝔅, yPr = impute!(𝔅, yPr, 𝓦𝓢, ll; verbose=verbose, it=i,
                                  solver=ST())

        accImpCounter += 1*acc

        if paramUpdt && i > warmUp
            for j in 1:updtLen
                (ll, acc, θ,
                 yPr) = updateParam!(updtType[j], 𝔅, tKernel, θ, updtCoord[j],
                                     yPr, 𝓦𝓢, ll, priors[j], recomputeODEs[j];
                                     solver=ST(), verbose=verbose, it=i, uidx=j)
                accUpdtCounter[j] += 1*acc
                updtStepCounter += 1
                θchain[updtStepCounter] = copy(θ)
                verbose && print("\n")
            end
            verbose && print("------------------------------------------------",
                             "------\n")
        end
        addPath!(adaptiveProp, 𝓦𝓢.XX, i)
        print_adaptation_info(adaptiveProp, accImpCounter, accUpdtCounter, i)
        adaptiveProp, 𝓦𝓢, yPr, ll = adaptationUpdt!(adaptiveProp, 𝓦𝓢, yPr, i,
                                                     ll, ObsScheme(), ST())
        adaptiveProp = still_adapting(adaptiveProp)
    end
    displayAcceptanceRate(𝔅)
    Time = collect(Iterators.flatten(p.tt[1:skipForSave:end-1] for p in P))
    θchain, accImpCounter/numSteps, accUpdtCounter./numSteps, Paths, Time
end
