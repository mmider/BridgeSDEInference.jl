using ForwardDiff: value

"""
    AbstractObsScheme

Types inheriting from abstract type `AbstractObsScheme` define the scheme
according to which a stochastic process has been observed
"""
abstract type AbstractObsScheme end


"""
    PartObs <: AbstractObsScheme

Type acting as a flag for partially observed diffusions
"""
struct PartObs <: AbstractObsScheme end


"""
    FPT <: AbstractObsScheme

Observation scheme in which only first passage times are observed
"""
struct FPT <: AbstractObsScheme end

"""
    ParamUpdateType

Types inheriting from abstract type `ParamUpdateType` define the way in which
parameters are to be updated by the MCMC sampler
"""
abstract type ParamUpdateType end

"""
    ConjugateUpdt <: ParamUpdateType

Type acting as a flag for update from full conditional (conjugate to a prior)
"""
struct ConjugateUpdt <: ParamUpdateType end

"""
    MetropolisHastingsUpdt <: ParamUpdateType

Flag for performing update according to Metropolis Hastings step
"""
struct MetropolisHastingsUpdt <: ParamUpdateType end


"""
    setBlocking(𝔅::NoBlocking, ::Any, ::Any, ::Any, ::Any)

No blocking is to be done, do nothing
"""
setBlocking(𝔅::NoBlocking, ::Any, ::Any, ::Any, ::Any) = 𝔅


"""
    setBlocking(::ChequeredBlocking, blockingParams, P, WW, XX)

Blocking pattern is chosen to be a chequerboard.
"""
function setBlocking(::ChequeredBlocking, blockingParams, P, WW, XX)
    ChequeredBlocking(blockingParams..., P, WW, XX)
end

"""
    FPTInfo{S,T}

The struct
```
struct FPTInfo{S,T}
    condCoord::NTuple{N,S}
    upCrossing::NTuple{N,Bool}
    autoRenewed::NTuple{N,Bool}
    reset::NTuple{N,T}
end
```
serves as a container for the information regarding first passage time
observations. `condCoord` is an NTuple of coordinates that are conditioned on
the first passage time nature of the observations. `upCrossing` indicates
whether observations of the corresponding coordinate are up-crossings or
down-crossings. `autoRenewed` indicates whether process starts from the
renewed state (i.e. normally the process is unconstrained until it hits level
`reset` for the first time, however `autoRenewed` process is constrained on the
first passage time from the very beginnig). `reset` level is the level that
needs to be reached before the process starts to be conditioned on the first
passage time.
"""
struct FPTInfo{S,T,N}
    condCoord::NTuple{N,S}
    upCrossing::NTuple{N,Bool}
    autoRenewed::NTuple{N,Bool}
    reset::NTuple{N,T}

    FPTInfo(condCoord::NTuple{N,S}, upCrossing::NTuple{N,Bool},
            reset::NTuple{N,T},
            autoRenewed::NTuple{N,Bool} = Tuple(fill(false,length(condCoord)))
            ) where {S,T,N} = new{S,T,N}(condCoord, upCrossing,
                                         autoRenewed, reset)
end


"""
    checkSingleCoordFpt(XXᵒ, c, cidx, fpt)

Verify whether coordinate `c` (with index number `cidx`) of path `XXᵒ`.yy
adheres to the first passage time observation scheme specified by the object
`fpt`.
"""
function checkSingleCoordFpt(XXᵒ, c, cidx, fpt)
    k = length(XXᵒ.yy)
    thrsd = XXᵒ.yy[end][c]
    renewed = fpt.autoRenewed[cidx]
    s = fpt.upCrossing[cidx] ? 1 : -1
    for i in 1:k
        if !renewed && (s*XXᵒ.yy[i][c] <= s*fpt.reset[cidx])
            renewed = true
        elseif renewed && (s*XXᵒ.yy[i][c] > s*thrsd)
            return false
        end
    end
    return renewed
end


"""
    checkFpt(::PartObs, XXᵒ, fpt)

First passage time constrains are automatically satisfied for the partially
observed scheme
"""
checkFpt(::PartObs, XXᵒ, fpt) = true


"""
    checkFpt(::FPT, XXᵒ, fpt)

Verify whether path `XXᵒ`.yy adheres to the first passage time observation
scheme specified by the object `fpt`.
"""
function checkFpt(::FPT, XXᵒ, fpt)
    for (cidx, c) in enumerate(fpt.condCoord)
        if !checkSingleCoordFpt(XXᵒ, c, cidx, fpt)
            return false
        end
    end
    return true
end


"""
    checkFullPathFpt(::PartObs, ::Any, ::Any, ::Any)

First passage time constrains are automatically satisfied for the partially
observed scheme
"""
checkFullPathFpt(::PartObs, ::Any, ::Any, ::Any) = true


"""
    checkFullPathFpt(::PartObs, XXᵒ, m, fpt)

Verify whether all paths in the range `iRange`, i.e. `XXᵒ`[i].yy, i in `iRange`
adhere to the first passage time observation scheme specified by the object
`fpt`
"""
function checkFullPathFpt(::FPT, XXᵒ, iRange, fpt)
    for i in iRange
        if !checkFpt(FPT(), XXᵒ[i], fpt[i])
            return false
        end
    end
    return true
end

"""
    checkDomainAdherence(P::Vector{ContinuousTimeProcess},
                         XX::Vector{SamplePath}, iRange)

Verify whether all paths in the range `iRange`, i.e. `XX[i].yy`, i in `iRange`
fall on the interior of the domain of diffusions `P[i]`, i in `iRange`
"""
function checkDomainAdherence(P::Vector{S}, XX::Vector{T}, iRange
                              ) where {S<:ContinuousTimeProcess, T<:SamplePath}
    for i in iRange
        !checkDomainAdherence(P[i], XX[i]) && return false
    end
    true
end

"""
    checkDomainAdherence(P::ContinuousTimeProcess, XX::SamplePath,
                         d::UnboundedDomain=domain(P))

For unrestricted domains there is nothing to check
"""
function checkDomainAdherence(P::ContinuousTimeProcess, XX::SamplePath,
                              d::UnboundedDomain=domain(P.Target))
    print("no restrictions...\n")
    true
end

"""
    checkDomainAdherence(P::ContinuousTimeProcess, XX::SamplePath,
                         d::DiffusionDomain=domain(P))

Verify whether path `XX.yy` falls on the interior of the domain of diffusion `P`
"""
function checkDomainAdherence(P::ContinuousTimeProcess, XX::SamplePath,
                              d::DiffusionDomain=domain(P.Target))
    N = length(XX)
    for i in 1:N
        !boundSatisfied(d, XX.yy[i]) && false
    end
    true
end


"""
    findProposalLaw(xx, tt, P˟, P̃, Ls, Σs; dt=1/5000, timeChange=true,
                    solver::ST=Ralston3())

Initialise the object with proposal law and all the necessary containers needed
for the simulation of the guided proposals
"""
function findProposalLaw(::Type{K}, xx, tt, P˟, P̃, Ls, Σs, τ; dt=1/5000,
                         solver::ST=Ralston3(),
                         changePt::ODEChangePt=NoChangePt()) where {K,ST}
    m = length(xx) - 1
    P = Array{ContinuousTimeProcess,1}(undef,m)
    for i in m:-1:1
        numPts = Int64(ceil((tt[i+1]-tt[i])/dt))+1
        t = τ(tt[i], tt[i+1]).( range(tt[i], stop=tt[i+1], length=numPts) )
        P[i] = ( (i==m) ? GuidPropBridge(K, t, P˟, P̃[i], Ls[i], xx[i+1], Σs[i];
                                         changePt=changePt, solver=ST()) :
                          GuidPropBridge(K, t, P˟, P̃[i], Ls[i], xx[i+1], Σs[i],
                                         P[i+1].H[1], P[i+1].Hν[1], P[i+1].c[1];
                                         changePt=changePt, solver=ST()) )
    end
    P
end


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


"""
    savePath!(Paths, XX, saveMe, skip)

If `saveMe` flag is true, then save the entire path spanning all segments in
`XX`. Only 1 in  every `skip` points is saved to reduce storage space.
"""
function savePath!(Paths, XX, saveMe, skip)
    if saveMe
        push!(Paths, collect(Iterators.flatten(XX[i].yy[1:skip:end-1]
                                               for i in 1:length(XX))))
    end
end


"""
    acceptSample(logThreshold, verbose=false)

Make a random MCMC decision for whether to accept a sample or reject it.
"""
function acceptSample(logThreshold, verbose=false)
    if rand(Exponential(1.0)) > -logThreshold # Reject if NaN
        verbose && print("\t ✓\n")
        return true
    else
        verbose && print("\t .\n")
        return false
    end
end


"""
    solveBackRec!(P, solver::ST=Ralston3()) where ST

Solve backward recursion to find H, Hν, c and Q, which together define r̃(t,x)
and p̃(x, 𝓓) under the auxiliary law, when no blocking is done
"""
function solveBackRec!(::NoBlocking, P, solver::ST=Ralston3()) where ST
    m = length(P)
    gpupdate!(P[m]; solver=ST())
    for i in (m-1):-1:1
        gpupdate!(P[i], P[i+1].H[1], P[i+1].Hν[1], P[i+1].c[1]; solver=ST())
    end
end


"""
    solveBackRec!(P, solver::ST=Ralston3()) where ST

Solve backward recursion to find H, Hν, c and Q, which together define r̃(t,x)
and p̃(x, 𝓓) under the auxiliary law, when blocking is done
"""
function solveBackRec!(𝔅::BlockingSchedule, P, solver::ST=Ralston3()) where ST
    for block in reverse(𝔅.blocks[𝔅.idx])
        gpupdate!(P[block[end]]; solver=ST())
        for i in reverse(block[1:end-1])
            gpupdate!(P[i], P[i+1].H[1], P[i+1].Hν[1], P[i+1].c[1]; solver=ST())
        end
    end
end

"""
    proposalStartPt(::BlockingSchedule, ::Val{1}, ::Any, yPr, P, ρ)

Set a new starting point for the proposal path when sampling the first block in
a blocking scheme.

...
# Arguments
- `::BlockingSchedule`: indicator that a blocking scheme is used
- `::Val{1}`: indicator that it's the first block, so starting point needs updating
- `yPr`: prior over the starting point
- `P`: diffusion law
- `ρ`: memory parameter in the Crank-Nicolson scheme
...
"""
function proposalStartPt(::BlockingSchedule, ::Val{1}, ::Any, yPr, P, ρ)
    proposalStartPt(NoBlocking(), nothing, nothing, yPr, P, ρ)
end

"""
    proposalStartPt(::BlockingSchedule, ::Any, y₀, yPr, ::Any, ::Any)

Default behaviour of dealing with a starting point in the blocking scheme is
to do nothing
"""
function proposalStartPt(::BlockingSchedule, ::Any, y₀, yPr, ::Any, ::Any)
    y₀, yPr
end

"""
    proposalStartPt(::NoBlocking, ::Any, y₀, yPr, P, ρ)

Set a new starting point for the proposal path when no blocking is done
...
# Arguments
- `::NoBlocking`: indicator that no blocking is done
- `yPr`: prior over the starting point
- `P`: diffusion law
- `ρ`: memory parameter in the Crank-Nicolson scheme
...
"""
function proposalStartPt(::NoBlocking, ::Any, ::Any, yPr, P, ρ)
    yPrᵒ = rand(yPr, ρ)
    y = startPt(yPrᵒ, P)
    y, yPrᵒ
end

"""
    printInfo(verbose::Bool, it::Integer, ll, llᵒ, msg="update")

Print information to the console about current likelihood values

...
# Arguments
- `verbose`: flag for whether to print anything at all
- `it`: iteration of the Markov chain
- `ll`: likelihood of the previous, accepted sample
- `llᵒ`: likelihood of the proposal sample
- `msg`: message to start with
...
"""
function printInfo(verbose::Bool, it::Integer, ll, llᵒ, msg="update")
    verbose && print(msg, ": ", it, " ll ", round(ll, digits=3), " ",
                     round(llᵒ, digits=3), " diff_ll: ", round(llᵒ-ll,digits=3))
end


"""
    pathLogLikhd(::ObsScheme, XX, P, iRange, fpt; skipFPT=false)

Compute likelihood for path `XX` to be observed under `P`. Only segments with
index numbers in `iRange` are considered. `fpt` contains relevant info about
checks regarding adherence to first passage time pattern. `skipFPT` if set to
`true` can skip the step of checking adherence to fpt pattern (used for
conjugate updates, or any updates that keep `XX` unchanged)
"""
function pathLogLikhd(::ObsScheme, XX, P, iRange, fpt; skipFPT=false
                      ) where ObsScheme <: AbstractObsScheme
    ll = 0.0
    for i in iRange
        ll += llikelihood(LeftRule(), XX[i], P[i])
    end
    !skipFPT && (ll = checkFullPathFpt(ObsScheme(), XX, iRange, fpt) ? ll : -Inf)
    !skipFPT && (ll += checkDomainAdherence(P, XX, iRange) ? 0.0 : -Inf)
    ll
end

"""
    swap!(A, Aᵒ, iRange)

Swap contents between containers A & Aᵒ in the index range iRange
"""
function swap!(A, Aᵒ, iRange)
    for i in iRange
        A[i], Aᵒ[i] = Aᵒ[i], A[i]
    end
end

"""
    swap!(A, Aᵒ, B, Bᵒ, iRange)

Swap contents between containers A & Aᵒ in the index range iRange, do the same
for containers B & Bᵒ
"""
function swap!(A, Aᵒ, B, Bᵒ, iRange)
    swap!(A, Aᵒ, iRange)
    swap!(B, Bᵒ, iRange)
end

"""
    crankNicolson!(yᵒ, y, ρ)

Preconditioned Crank Nicolson update with memory parameter `ρ`, previous vector
`y` and new vector `yᵒ`
"""
crankNicolson!(yᵒ, y, ρ) = (yᵒ .= √(1-ρ)*yᵒ + √(ρ)*y)


"""
    sampleSegment!(i, Wnr, WW, WWᵒ, P, y, XX, ρ)

Sample `i`th path segment using preconditioned Crank-Nicolson scheme
...
# Arguments
- `i`: index of the segment to be sampled
- `Wnr`: type of the Wiener process
- `WW`: containers with old Wiener paths
- `WWᵒ`: containers where proposal Wiener paths will be stored
- `P`: laws of the diffusion to be sampled
- `y`: starting point of the segment
- `XX`: containers for proposal diffusion path
- `ρ`: memory parameter for the Crank-Nicolson scheme
...
"""
function sampleSegment!(i, Wnr, WW, WWᵒ, P, y, XX, ρ)
    sample!(WWᵒ[i], Wnr)
    crankNicolson!(WWᵒ[i].yy, WW[i].yy, ρ)
    solve!(Euler(), XX[i], y, WWᵒ[i], P[i])
    XX[i].yy[end]
end


"""
    sampleSegments!(iRange, Wnr, WW, WWᵒ, P, y, XX, ρ)

Sample paths segments in index range `iRange` using preconditioned
Crank-Nicolson scheme
...
# Arguments
- `iRange`: range of indices of the segments that need to be sampled
- `Wnr`: type of the Wiener process
- `WW`: containers with old Wiener paths
- `WWᵒ`: containers where proposal Wiener paths will be stored
- `P`: laws of the diffusion to be sampled
- `y`: starting point of the segment
- `XX`: containers for proposal diffusion path
- `ρ`: memory parameter for the Crank-Nicolson scheme
...
"""
function sampleSegments!(iRange, Wnr, WW, WWᵒ, P, y, XX, ρ)
    for i in iRange
        y = sampleSegment!(i, Wnr, WW, WWᵒ, P, y, XX, ρ)
    end
end


"""
    impute!(::ObsScheme, 𝔅::NoBlocking, Wnr, yPr, WWᵒ, WW, XXᵒ, XX, P, ll, fpt;
            ρ=0.0, verbose=false, it=NaN, headStart=false) where
            ObsScheme <: AbstractObsScheme -> acceptedLogLikhd, acceptDecision

Imputation step of the MCMC scheme (without blocking).
...
# Arguments
- `::ObsScheme`: observation scheme---first-passage time or partial observations
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
function impute!(::ObsScheme, 𝔅::NoBlocking, Wnr, yPr, WWᵒ, WW, XXᵒ, XX, P, ll,
                 fpt; ρ=0.0, verbose=false, it=NaN, headStart=false,
                 solver::ST=Ralston3()) where
                 {ObsScheme <: AbstractObsScheme, ST}
    # sample proposal starting point
    yᵒ, yPrᵒ = proposalStartPt(𝔅, nothing, nothing, yPr, P[1], ρ)

    # sample proposal path
    m = length(WWᵒ)
    yᵗᵉᵐᵖ = copy(yᵒ)
    for i in 1:m
        sampleSegment!(i, Wnr, WW, WWᵒ, P, yᵗᵉᵐᵖ, XXᵒ, ρ)
        if headStart
            while !checkFpt(ObsScheme(), XXᵒ[i], fpt[i])
                sampleSegment!(i, Wnr, WW, WWᵒ, P, yᵗᵉᵐᵖ, XXᵒ, ρ)
            end
        end
        yᵗᵉᵐᵖ = XXᵒ[i].yy[end]
    end

    llᵒ = logpdf(yPrᵒ, yᵒ)
    llᵒ += pathLogLikhd(ObsScheme(), XXᵒ, P, 1:m, fpt)
    llᵒ += lobslikelihood(P[1], yᵒ)

    printInfo(verbose, it, value(ll), value(llᵒ), "impute")

    if acceptSample(llᵒ-ll, verbose)
        swap!(XX, XXᵒ, WW, WWᵒ, 1:m)
        return llᵒ, true, 𝔅, yPrᵒ
    else
        return ll, false, 𝔅, yPr
    end
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


"""
    noiseFromPath!(𝔅::BlockingSchedule, XX, WW, P)

Compute driving Wiener noise `WW` from path `XX` drawn under law `P`
"""
function noiseFromPath!(𝔅::BlockingSchedule, XX, WW, P)
    for block in 𝔅.blocks[𝔅.idx]
        for i in block
            invSolve!(Euler(), XX[i], WW[i], P[i])
        end
    end
end


"""
    startPtLogPdf(::Val{1}, yPr::StartingPtPrior, y)

Compute the log-likelihood contribution of the starting point for a given prior
under a blocking scheme (intended to be used with a first block only)
"""
startPtLogPdf(::Val{1}, yPr::StartingPtPrior, y) = logpdf(yPr, y)

"""
    startPtLogPdf(::Any, yPr::StartingPtPrior, y)

Default contribution to log-likelihood from the startin point under blocking
"""
startPtLogPdf(::Any, yPr::StartingPtPrior, y) = 0.0


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
function impute!(::ObsScheme, 𝔅::ChequeredBlocking, Wnr, yPr, WWᵒ, WW, XXᵒ, XX,
                 P, ll, fpt; ρ=0.0, verbose=false, it=NaN, headStart=false,
                 solver::ST=Ralston3()
                 ) where {ObsScheme <: AbstractObsScheme, ST}
    θ = params(𝔅.P[1].Target)             # current parameter
    𝔅 = next(𝔅, 𝔅.XX, θ)
    solveBackRec!(𝔅, 𝔅.P, ST())         # compute (H, Hν, c) for given blocks
    noiseFromPath!(𝔅, 𝔅.XX, 𝔅.WW, 𝔅.P) # find noise WW that generates XX under 𝔅.P

    # compute white noise generating starting point under 𝔅
    yPr = invStartPt(𝔅.XX[1].yy[1], yPr, 𝔅.P[1])

    ll_total = 0.0
    for (blockIdx, block) in enumerate(𝔅.blocks[𝔅.idx])
        blockFlag = Val{block[1]}()
        y = 𝔅.XX[block[1]].yy[1]       # accepted starting point

        # proposal starting point for the block (can be non-y only for the first block)
        yᵒ, yPrᵒ = proposalStartPt(𝔅, blockFlag, y, yPr, 𝔅.P[block[1]], ρ)

        # sample path in block
        sampleSegments!(block, Wnr, 𝔅.WW, 𝔅.WWᵒ, 𝔅.P , yᵒ, 𝔅.XXᵒ, ρ)
        setEndPtManually!(𝔅, blockIdx, block)

        # starting point, path and observations contribution
        llᵒ = startPtLogPdf(blockFlag, yPrᵒ, yᵒ)
        llᵒ += pathLogLikhd(ObsScheme(), 𝔅.XXᵒ, 𝔅.P, block, fpt)
        llᵒ += lobslikelihood(𝔅.P[block[1]], yᵒ)

        llPrev = startPtLogPdf(blockFlag, yPr, y)
        llPrev += pathLogLikhd(ObsScheme(), 𝔅.XX, 𝔅.P, block, fpt; skipFPT=true)
        llPrev += lobslikelihood(𝔅.P[block[1]], y)

        printInfo(verbose, it, value(llPrev), value(llᵒ), "impute")
        if acceptSample(llᵒ-llPrev, verbose)
            swap!(𝔅.XX, 𝔅.XXᵒ, block)
            registerAccpt!(𝔅, blockIdx, true)
            yPr = yPrᵒ # can do something non-trivial only for the first block
            ll_total += llᵒ
        else
            registerAccpt!(𝔅, blockIdx, false)
            ll_total += llPrev
        end
    end
    # acceptance indicator does not matter for sampling with blocking
    return ll_total, true, 𝔅, yPr
end

"""
    updateLaws!(Ps, θᵒ)

Set new parameter `θᵒ` for the laws in vector `Ps`
"""
function updateLaws!(Ps, θᵒ)
    m = length(Ps)
    for i in 1:m
        Ps[i] = GuidPropBridge(Ps[i], θᵒ)
    end
end

"""
    updateTargetLaws!(𝔅::NoBlocking, θᵒ)

Nothing to do
"""
updateTargetLaws!(𝔅::NoBlocking, θᵒ) = nothing

"""
    updateTargetLaws!(𝔅::BlockingSchedule, θᵒ)

Set new parameter `θᵒ` for the target laws in blocking object `𝔅`
"""
function updateTargetLaws!(𝔅::BlockingSchedule, θᵒ)
    for block in 𝔅.blocks[𝔅.idx]
        for i in block
            𝔅.P[i] = GuidPropBridge(𝔅.P[i], θᵒ)
        end
    end
end

"""
    updateProposalLaws!(𝔅::BlockingSchedule, θᵒ)

Set new parameter `θᵒ` for the proposal laws inside blocking object `𝔅`
"""
function updateProposalLaws!(𝔅::BlockingSchedule, θᵒ)
    for block in 𝔅.blocks[𝔅.idx]
        for i in block
            𝔅.Pᵒ[i] = GuidPropBridge(𝔅.Pᵒ[i], θᵒ)
        end
    end
end

"""
    findPathFromWiener!(XX, y, WW, P, iRange)

Find path `XX` (that starts from `y`) that is generated under law `P` from the
Wiener process `WW`. Only segments with indices in range `iRange` are considered
"""
function findPathFromWiener!(XX, y, WW, P, iRange)
    for i in iRange
        solve!(Euler(), XX[i], y, WW[i], P[i])
        y = XX[i].yy[end]
    end
end


"""
    priorKernelContrib(tKern, priors, θ, θᵒ)

Contribution to the log-likelihood ratio from transition kernel `tKernel` and
`priors`.
"""
function priorKernelContrib(tKern, priors, θ, θᵒ)
    llr = logpdf(tKern, θᵒ, θ) - logpdf(tKern, θ, θᵒ)
    for prior in priors
        llr += logpdf(prior, θᵒ) - logpdf(prior, θ)
    end
    llr
end


"""
    setEndPtManually!(𝔅::BlockingSchedule, blockIdx, block)

Manually set the end-point of the proposal path under blocking so that it agrees
with the end-point of the previously accepted path. If it is the last block,
then do nothing
"""
function setEndPtManually!(𝔅::BlockingSchedule, blockIdx, block)
    if blockIdx < length(𝔅.blocks[𝔅.idx])
        𝔅.XXᵒ[block[end]].yy[end] = 𝔅.XX[block[end]].yy[end]
    end
end


"""
    updateParam!(::ObsScheme, ::MetropolisHastingsUpdt, tKern, θ, ::UpdtIdx,
                 yPr, WW, Pᵒ, P, XXᵒ, XX, ll, prior, fpt, recomputeODEs;
                 solver::ST=Ralston3(), verbose=false,
                 it=NaN) where {ObsScheme <: AbstractObsScheme, ST, UpdtIdx}
                 -> acceptedLogLikhd, acceptDecision
Update parameters
...
# Arguments
- `::ObsScheme`: observation scheme---first-passage time or partial observations
- `::MetropolisHastingsUpdt()`: type of the parameter update
- `tKern`: transition kernel
- `θ`: current value of the parameter
- `updtIdx`: object declaring indices of the updated parameter
- `yPr`: prior over the starting point of the diffusion path
- `WW`: containers with Wiener paths
- `Pᵒ`: container for the laws of the diffusion path with new parametrisation
- `P`: laws of the diffusion path with old parametrisation
- `XXᵒ`: containers for proposal diffusion paths
- `XX`: containers with old diffusion paths
- `11`: likelihood of the old (previously accepted) parametrisation
- `priors`: list of priors
- `fpt`: info about first-passage time conditioning
- `recomputeODEs`: whether auxiliary law depends on the updated params
- `verbose`: whether to print updates info while sampling
- `it`: iteration index of the MCMC algorithm
...
"""
function updateParam!(::ObsScheme, ::MetropolisHastingsUpdt, 𝔅::NoBlocking,
                      tKern, θ, ::UpdtIdx, yPr, WW, Pᵒ, P, XXᵒ, XX, ll, priors,
                      fpt, recomputeODEs; solver::ST=Ralston3(), verbose=false,
                      it=NaN) where {ObsScheme <: AbstractObsScheme, ST, UpdtIdx}
    m = length(WW)
    θᵒ = rand(tKern, θ, UpdtIdx())               # sample new parameter
    updateLaws!(Pᵒ, θᵒ)
    recomputeODEs && solveBackRec!(NoBlocking(), Pᵒ, ST()) # compute (H, Hν, c)

    # find white noise which for a given θᵒ gives a correct starting point
    y = XX[1].yy[1]
    yPrᵒ = invStartPt(y, yPr, Pᵒ[1])

    findPathFromWiener!(XXᵒ, y, WW, Pᵒ, 1:m)

    llᵒ = logpdf(yPrᵒ, y)
    llᵒ += pathLogLikhd(ObsScheme(), XXᵒ, Pᵒ, 1:m, fpt)
    llᵒ += lobslikelihood(Pᵒ[1], y)

    printInfo(verbose, it, ll, llᵒ)

    llr = ( llᵒ - ll + priorKernelContrib(tKern, priors, θ, θᵒ))

    # Accept / reject
    if acceptSample(llr, verbose)
        swap!(XX, XXᵒ, P, Pᵒ, 1:m)
        return llᵒ, true, θᵒ, yPrᵒ
    else
        return ll, false, θ, yPr
    end
end


"""
    updateParam!(::ObsScheme, ::MetropolisHastingsUpdt, tKern, θ, ::UpdtIdx,
                 yPr, WW, Pᵒ, P, XXᵒ, XX, ll, prior, fpt, recomputeODEs;
                 solver::ST=Ralston3(), verbose=false,
                 it=NaN) where {ObsScheme <: AbstractObsScheme, ST, UpdtIdx}
                 -> acceptedLogLikhd, acceptDecision
Update parameters
...
# Arguments
- `::ObsScheme`: observation scheme---first-passage time or partial observations
- `::MetropolisHastingsUpdt()`: type of the parameter update
- `tKern`: transition kernel
- `θ`: current value of the parameter
- `updtIdx`: object declaring indices of the updated parameter
- `yPr`: prior over the starting point of the diffusion path
- `WW`: containers with Wiener paths
- `Pᵒ`: container for the laws of the diffusion path with new parametrisation
- `P`: laws of the diffusion path with old parametrisation
- `XXᵒ`: containers for proposal diffusion paths
- `XX`: containers with old diffusion paths
- `11`: likelihood of the old (previously accepted) parametrisation
- `priors`: list of priors
- `fpt`: info about first-passage time conditioning
- `recomputeODEs`: whether auxiliary law depends on the updated params
- `verbose`: whether to print updates info while sampling
- `it`: iteration index of the MCMC algorithm
...
"""
function updateParam!(::ObsScheme, ::MetropolisHastingsUpdt,
                      𝔅::ChequeredBlocking, tKern, θ, ::UpdtIdx,
                      yPr, WW, Pᵒ, P, XXᵒ, XX, ll, priors, fpt, recomputeODEs;
                      solver::ST=Ralston3(), verbose=false,
                      it=NaN) where {ObsScheme <: AbstractObsScheme, ST, UpdtIdx}
    m = length(WW)
    θᵒ = rand(tKern, θ, UpdtIdx())               # sample new parameter
    updateProposalLaws!(𝔅, θᵒ)                   # update law `Pᵒ` accordingly
    solveBackRec!(𝔅, 𝔅.Pᵒ, ST())                 # compute (H, Hν, c)

    llᵒ = logpdf(yPr, 𝔅.XX[1].yy[1])
    for (blockIdx, block) in enumerate(𝔅.blocks[𝔅.idx])
        y = 𝔅.XX[block[1]].yy[1]
        findPathFromWiener!(𝔅.XXᵒ, y, 𝔅.WW, 𝔅.Pᵒ, block)
        setEndPtManually!(𝔅, blockIdx, block)

        # Compute log-likelihood ratio
        llᵒ += pathLogLikhd(ObsScheme(), 𝔅.XXᵒ, 𝔅.Pᵒ, block, fpt)
        llᵒ += lobslikelihood(𝔅.Pᵒ[block[1]], y)
    end
    printInfo(verbose, it, ll, llᵒ)

    llr = ( llᵒ - ll + priorKernelContrib(tKern, priors, θ, θᵒ))

    # Accept / reject
    if acceptSample(llr, verbose)
        swap!(𝔅.XX, 𝔅.XXᵒ, 𝔅.P, 𝔅.Pᵒ, 1:m)
        return llᵒ, true, θᵒ, yPr
    else
        return ll, false, θ, yPr
    end
end


fetchTargetLaw(𝔅::NoBlocking, P) = P[1].Target

fetchTargetLaw(𝔅::BlockingSchedule, P) = 𝔅.P[1].Target


"""
    updateParam!(::PartObs, ::ConjugateUpdt, tKern, θ, ::UpdtIdx, yPr, WW, Pᵒ,
                 P, XXᵒ, XX, ll, priors, fpt, recomputeODEs;
                 solver::ST=Ralston3(), verbose=false, it=NaN
                 ) -> acceptedLogLikhd, acceptDecision
Update parameters
see the definition of  updateParam!(…, ::MetropolisHastingsUpdt, …) for the
explanation of the arguments.
"""
function updateParam!(::ObsScheme, ::ConjugateUpdt, 𝔅::NoBlocking,
                      tKern, θ, ::UpdtIdx, yPr, WW, Pᵒ, P, XXᵒ, XX, ll, priors,
                      fpt, recomputeODEs; solver::ST=Ralston3(), verbose=false,
                      it=NaN) where {ObsScheme <: AbstractObsScheme, ST, UpdtIdx}
    m = length(P)
    ϑ = conjugateDraw(θ, XX, P[1].Target, priors[1], UpdtIdx())   # sample new parameter
    θᵒ = moveToProperPlace(ϑ, θ, UpdtIdx())     # align so that dimensions agree

    updateLaws!(P, θᵒ)
    recomputeODEs && solveBackRec!(NoBlocking(), P, ST()) # compute (H, Hν, c)

    for i in 1:m    # compute wiener path WW that generates XX
        invSolve!(Euler(), XX[i], WW[i], P[i])
    end
    # compute white noise that generates starting point
    y = XX[1].yy[1]
    yPr = invStartPt(y, yPr, P[1])

    llᵒ = logpdf(yPr, y)
    llᵒ += pathLogLikhd(ObsScheme(), XX, P, 1:m, fpt; skipFPT=true)
    llᵒ += lobslikelihood(P[1], y)
    printInfo(verbose, it, value(ll), value(llᵒ))
    return llᵒ, true, θᵒ, yPr
end


"""
    updateParam!(::PartObs, ::ConjugateUpdt, tKern, θ, ::UpdtIdx, yPr, WW, Pᵒ,
                 P, XXᵒ, XX, ll, priors, fpt, recomputeODEs;
                 solver::ST=Ralston3(), verbose=false, it=NaN
                 ) -> acceptedLogLikhd, acceptDecision
Update parameters
see the definition of  updateParam!(…, ::MetropolisHastingsUpdt, …) for the
explanation of the arguments.
"""
function updateParam!(::ObsScheme, ::ConjugateUpdt, 𝔅::BlockingSchedule,
                      tKern, θ, ::UpdtIdx, yPr, WW, Pᵒ, P, XXᵒ, XX, ll, priors,
                      fpt, recomputeODEs; solver::ST=Ralston3(), verbose=false,
                      it=NaN) where {ObsScheme <: AbstractObsScheme, ST, UpdtIdx}
    m = length(P)
    ϑ = conjugateDraw(θ, 𝔅.XX, 𝔅.P[1].Target, priors[1], UpdtIdx())   # sample new parameter
    θᵒ = moveToProperPlace(ϑ, θ, UpdtIdx())     # align so that dimensions agree

    updateTargetLaws!(𝔅, θᵒ)
    recomputeODEs && solveBackRec!(𝔅, 𝔅.P, ST())
    for i in 1:m    # compute wiener path WW that generates XX
        invSolve!(Euler(), 𝔅.XX[i], 𝔅.WW[i], 𝔅.P[i])
    end
    # compute white noise that generates starting point
    y = 𝔅.XX[1].yy[1]
    yPr = invStartPt(y, yPr, 𝔅.P[1])
    llᵒ = logpdf(yPr, y)
    for block in 𝔅.blocks[𝔅.idx]
        llᵒ += pathLogLikhd(ObsScheme(), 𝔅.XX, 𝔅.P, block, fpt; skipFPT=true)
        llᵒ += lobslikelihood(𝔅.P[block[1]], 𝔅.XX[block[1]].yy[1])
    end
    printInfo(verbose, it, value(ll), value(llᵒ))
    return llᵒ, true, θᵒ, yPr
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
              solver::ST=Ralston3(), changePt::CP=NoChangePt(), warmUp=0
              ) where {K, ObsScheme <: AbstractObsScheme, ST, Blocking, CP}
    P = findProposalLaw( K, obs, obsTimes, P˟, P̃, Ls, Σs, τ; dt=dt, solver=ST(),
                         changePt=CP(getChangePt(blockingParams[3])) )
    m = length(obs)-1
    updtLen = length(updtCoord)
    Wnr, WWᵒ, WW, XXᵒ, XX, Pᵒ, ll, yPr = initialise(ObsScheme(), P, m, yPr, w,
                                                    fpt)
    Paths = []
    accImpCounter = 0
    accUpdtCounter = [0 for i in 1:updtLen]
    θ = params(P˟)
    θchain = Vector{typeof(θ)}(undef, (numSteps-warmUp)*updtLen+1)
    θchain[1] = copy(θ)
    recomputeODEs = [any([e in dependsOnParams(P[1].Pt) for e
                         in idx(uc)]) for uc in updtCoord]

    updtStepCounter = 1
    𝔅 = setBlocking(blocking, blockingParams, P, WW, XX)
    display(𝔅)
    for i in 1:numSteps
        verbose = (i % verbIter == 0)
        i > warmUp && savePath!(Paths, blocking == NoBlocking() ? XX : 𝔅.XX,
                                (i % saveIter == 0), skipForSave)
        ll, acc, 𝔅, yPr = impute!(ObsScheme(), 𝔅, Wnr, yPr, WWᵒ, WW, XXᵒ, XX,
                                  P, ll, fpt, ρ=ρ, verbose=verbose, it=i,
                                  solver=ST())
        accImpCounter += 1*acc
        if paramUpdt && i > warmUp
            for j in 1:updtLen
                (ll, acc, θ,
                 yPr) = updateParam!(ObsScheme(), updtType[j], 𝔅, tKernel, θ,
                                     updtCoord[j], yPr, WW, Pᵒ, P, XXᵒ, XX, ll,
                                     priors[j], fpt, recomputeODEs[j];
                                     solver=ST(), verbose=verbose, it=i)
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
