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
