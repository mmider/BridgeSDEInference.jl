#NOTE this import is bad programming, will need to change
import Main: clone, conjugateDraw, dependsOnParams, params

#dependsOnParams(::ContinuousTimeProcess) = (,)

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


setBlocking(𝔅::NoBlocking, ::Any, ::Any, ::Any, ::Any) = 𝔅

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
    if fpt.upCrossing[cidx]
        for i in 1:k
            if !renewed && XXᵒ.yy[i][c] <= fpt.reset[cidx]
                renewed = true
            elseif renewed && XXᵒ.yy[i][c] > thrsd
                return false
            end
        end
    else
        for i in 1:k
            if !renewed && XXᵒ.yy[i][c] >= fpt.reset[cidx]
                renewed = true
            elseif renewed && XXᵒ.yy[i][c] < thrsd
                return false
            end
        end
    end
    return true
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
    checkFullPathFpt(::PartObs, XXᵒ, m, fpt)

First passage time constrains are automatically satisfied for the partially
observed scheme
"""
checkFullPathFpt(::PartObs, XXᵒ, m, fpt) = true


"""
    checkFullPathFpt(::PartObs, XXᵒ, m, fpt)

Verify whether all `m` paths `XXᵒ`[i].yy, i=1,...,m adhere to the first passage
time observation scheme specified by the object `fpt`.
"""
function checkFullPathFpt(::FPT, XXᵒ, m, fpt)
    for i in 1:m
        if !checkFpt(FPT(), XXᵒ[i], fpt[i])
            return false
        end
    end
    return true
end

"""
    findProposalLaw(xx, tt, P˟, P̃, Ls, Σs; dt=1/5000, timeChange=true,
                    solver::ST=Ralston3())

Initialise the object with proposal law and all the necessary containers needed
for the simulation of the guided proposals
"""
function findProposalLaw(::Type{K}, xx, tt, P˟, P̃, Ls, Σs, τ; dt=1/5000,
                         solver::ST=Ralston3()) where {K,ST}
    m = length(xx) - 1
    P = Array{ContinuousTimeProcess,1}(undef,m)
    for i in m:-1:1
        numPts = Int64(ceil((tt[i+1]-tt[i])/dt))+1
        t = τ(tt[i], tt[i+1]).( range(tt[i], stop=tt[i+1], length=numPts) )
        P[i] = ( (i==m) ? GuidPropBridge(K, t, P˟, P̃[i], Ls[i], xx[i+1], Σs[i];
                                         solver=ST()) :
                          GuidPropBridge(K, t, P˟, P̃[i], Ls[i], xx[i+1], Σs[i],
                                         P[i+1].H[1], P[i+1].Hν[1], P[i+1].c[1],
                                         P[i+1].Q[1]; solver=ST()) )
    end
    P
end


"""
    initialise(::ObsScheme, P, m, y::T, ::S, fpt)

Initialise the workspace for MCMC algorithm. Initialises containers for driving
Wiener processes `WWᵒ` & `WW`, for diffusion processes `XXᵒ` & `XX`, for
diffusion Law `Pᵒ` (parametetrised by proposal parameters) and defines the type
of Wiener process `Wnr`.
"""
function initialise(::ObsScheme, P, m, y::T, ::S,
                    fpt) where {ObsScheme <: AbstractObsScheme,T,S}
    Pᵒ = deepcopy(P)
    TW = typeof(sample([0], Wiener{S}()))
    TX = typeof(SamplePath([], zeros(T, 0)))
    XXᵒ = Vector{TX}(undef,m)
    WWᵒ = Vector{TW}(undef,m)
    Wnr = Wiener{S}()
    ll = 0.0
    for i in 1:m
        WWᵒ[i] = Bridge.samplepath(P[i].tt, zero(S))
        sample!(WWᵒ[i], Wnr)
        XXᵒ[i] = solve(Euler(), y, WWᵒ[i], P[i])
        while !checkFpt(ObsScheme(), XXᵒ[i], fpt[i])
            sample!(WWᵒ[i], Wnr)
            solve!(Euler(), XXᵒ[i], y, WWᵒ[i], P[i])
        end
        y = XXᵒ[i].yy[end]
        ll += llikelihood(LeftRule(), XXᵒ[i], P[i])
    end
    XX = deepcopy(XXᵒ)
    WW = deepcopy(WWᵒ)
    Wnr, WWᵒ, WW, XXᵒ, XX, Pᵒ, ll
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
    if rand(Exponential(1.0)) > -logThreshold
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
and p̃(x, 𝓓) under the auxiliary law
"""
function solveBackRec!(P, solver::ST=Ralston3()) where ST
    m = length(P)
    gpupdate!(P[m]; solver=ST())
    for i in (m-1):-1:1
        gpupdate!(P[i], P[i+1].H[1], P[i+1].Hν[1], P[i+1].c[1],
                  P[i+1].Q[1]; solver=ST())
    end
end


"""
    impute!(::ObsScheme, Wnr, y, WWᵒ, WW, XXᵒ, XX, P, ll, fpt;
            ρ=0.0, verbose=false, it=NaN, headStart=false) where
            ObsScheme <: AbstractObsScheme -> acceptedLogLikhd, acceptDecision

Imputation step of the MCMC scheme.
...
# Arguments
- `::ObsScheme`: observation scheme---first-passage time or partial observations
- `Wnr`: type of the Wiener process
- `y`: starting point of the diffusion path
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
function impute!(::ObsScheme, Wnr, y, WWᵒ, WW, XXᵒ, XX, P, ll, fpt;
                 ρ=0.0, verbose=false, it=NaN, headStart=false) where
                 ObsScheme <: AbstractObsScheme
    m = length(WWᵒ)
    for i in 1:m
        sample!(WWᵒ[i], Wnr)
        WWᵒ[i].yy .= sqrt(1-ρ)*WWᵒ[i].yy + sqrt(ρ)*WW[i].yy
        solve!(Euler(),XXᵒ[i], y, WWᵒ[i], P[i])
        if headStart
            while !checkFpt(ObsScheme(), XXᵒ[i], fpt[i])
                sample!(WWᵒ[i], Wnr)
                WWᵒ[i].yy .= sqrt(1-ρ)*WWᵒ[i].yy + sqrt(ρ)*WW[i].yy
                solve!(Euler(), XXᵒ[i], y, WWᵒ[i], P[i])
            end
        end
        y = XXᵒ[i].yy[end]
    end

    llᵒ = 0.0
    for i in 1:m
        llᵒ += llikelihood(LeftRule(), XXᵒ[i], P[i])
    end
    llᵒ = checkFullPathFpt(ObsScheme(), XXᵒ, m, fpt) ? llᵒ : -Inf

    verbose && print("impute: ", it, " ll ", round(value(ll), digits=3), " ",
                     round(value(llᵒ), digits=3), " diff_ll: ", round(value(llᵒ-ll),digits=3))
    if acceptSample(llᵒ-ll, verbose)
        for i in 1:m
            XX[i], XXᵒ[i] = XXᵒ[i], XX[i]
            WW[i], WWᵒ[i] = WWᵒ[i], WW[i]
        end
        return llᵒ, true
    else
        return ll, false
    end
end


"""
    updateParam!(::ObsScheme, ::MetropolisHastingsUpdt, tKern, θ, ::UpdtIdx, y,
                 WW, Pᵒ, P, XXᵒ, XX, ll, prior, fpt, recomputeODEs;
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
- `y`: starting point of the diffusion path
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
function updateParam!(::ObsScheme, ::MetropolisHastingsUpdt, tKern, θ, ::UpdtIdx,
                      y, WW, Pᵒ, P, XXᵒ, XX, ll, priors, fpt, recomputeODEs;
                      solver::ST=Ralston3(), verbose=false,
                      it=NaN) where {ObsScheme <: AbstractObsScheme, ST, UpdtIdx}
    m = length(WW)
    θᵒ = rand(tKern, θ, UpdtIdx())
    for i in 1:m
        Pᵒ[i] = GuidPropBridge(Pᵒ[i], θᵒ)
    end
    recomputeODEs && solveBackRec!(Pᵒ, ST())

    y₀ = copy(y)
    for i in 1:m
        solve!(Euler(), XXᵒ[i], y₀, WW[i], Pᵒ[i])
        y₀ = XXᵒ[i].yy[end]
    end

    llᵒ = 0.0
    for i in 1:m
        llᵒ += llikelihood(LeftRule(), XXᵒ[i], Pᵒ[i])
    end
    llᵒ = checkFullPathFpt(ObsScheme(), XXᵒ, m, fpt) ? llᵒ : -Inf
    verbose && print("update: ", it, " ll ", round(ll, digits=3), " ",
                     round(llᵒ, digits=3), " diff_ll: ", round(llᵒ-ll,digits=3))
    llr = ( llᵒ - ll + logpdf(tKern, θᵒ, θ) - logpdf(tKern, θ, θᵒ) )
    for prior in priors
        llr += logpdf(prior, θᵒ) - logpdf(prior, θ)
    end
    recomputeODEs && (llr += lobslikelihood(Pᵒ[1], y) - lobslikelihood(P[1], y))
    if acceptSample(llr, verbose)
        for i in 1:m
            XX[i], XXᵒ[i] = XXᵒ[i], XX[i]
            P[i], Pᵒ[i] = Pᵒ[i], P[i]
        end
        return llᵒ, true, θᵒ
    else
        return ll, false, θ
    end
end


# NOTE it should work with FPT as well with additional rejection step
# but the practicality of this is untested, so supports only ::PartObs
# for now
"""
    updateParam!(::PartObs, ::ConjugateUpdt, tKern, θ, ::UpdtIdx, y, WW, Pᵒ, P,
                 XXᵒ, XX, ll, priors, fpt, recomputeODEs; solver::ST=Ralston3(),
                 verbose=false, it=NaN) -> acceptedLogLikhd, acceptDecision
Update parameters
see the definition of  updateParam!(…, ::MetropolisHastingsUpdt, …) for the
explanation of the arguments.
"""
function updateParam!(::PartObs, ::ConjugateUpdt, tKern, θ, ::UpdtIdx,
                      y, WW, Pᵒ, P, XXᵒ, XX, ll, priors, fpt, recomputeODEs;
                      solver::ST=Ralston3(), verbose=false,
                      it=NaN) where {ObsScheme <: AbstractObsScheme, ST, UpdtIdx}
    m = length(P)
    ϑ = conjugateDraw(θ, XX, P, priors[1], UpdtIdx())
    θᵒ = moveToProperPlace(ϑ, θ, UpdtIdx())

    for i in 1:m
        P[i] = GuidPropBridge(P[i], θᵒ)
    end
    recomputeODEs && solveBackRec!(P, ST())

    for i in 1:m
        solve!(Euler(), XX[i], y, WW[i], P[i])
        y = XX[i].yy[end]
    end

    llᵒ = 0.0
    for i in 1:m
        llᵒ += llikelihood(LeftRule(), XX[i], P[i])
    end
    verbose && print("update: ", it, " ll ", round(value(ll), digits=3), " ",
                     round(value(llᵒ), digits=3), " diff_ll: ",
                     round(value(llᵒ-ll),digits=3), "\n")
    return llᵒ, true, θᵒ
end

"""
    mcmc(::ObsScheme, obs, obsTimes, y, w, P˟, P̃, Ls, Σs, numSteps, tKernel,
         priors; fpt=fill(NaN, length(obsTimes)-1), ρ=0.0, dt=1/5000,
         timeChange=true, saveIter=NaN, verbIter=NaN,
         updtCoord=(Val((true,)),), paramUpdt=true, skipForSave=1,
         updtType=(MetropolisHastingsUpdt(),), solver::ST=Ralston3())

Gibbs sampler alternately imputing unobserved parts of the path and updating
unknown coordinates of the parameter vector (the latter only if paramUpdt==true)
...
# Arguments
- `::ObsScheme`: observation scheme---first-passage time or partial observations
- `obs`: vector with observations
- `obsTimes`: times of the observations
- `y`: starting point of the diffusion path
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
...
"""
function mcmc(::Type{K}, ::ObsScheme, obs, obsTimes, y, w, P˟, P̃, Ls, Σs, numSteps,
              tKernel, priors, τ; fpt=fill(NaN, length(obsTimes)-1), ρ=0.0,
              dt=1/5000, saveIter=NaN, verbIter=NaN,
              updtCoord=(Val((true,)),), paramUpdt=true,
              skipForSave=1, updtType=(MetropolisHastingsUpdt(),),
              blocking::Blocking=NoBlocking(), blockingParams=([], 0.1),
              solver::ST=Ralston3()) where {K, ObsScheme <: AbstractObsScheme, ST, Blocking}
    P = findProposalLaw(K, obs, obsTimes, P˟, P̃, Ls, Σs, τ; dt=dt, solver=ST())
    m = length(obs)-1
    updtLen = length(updtCoord)
    Wnr, WWᵒ, WW, XXᵒ, XX, Pᵒ, ll = initialise(ObsScheme(), P, m, y, w, fpt)
    Paths = []
    accImpCounter = 0
    accUpdtCounter = [0 for i in 1:updtLen]
    θ = params(P˟)
    θchain = Vector{typeof(θ)}(undef, numSteps+1)
    θchain[1] = copy(θ)

    𝔅 = setBlocking(blocking, blockingParams, P, WW, XX)
    display(𝔅)

    for i in 1:numSteps
        verbose = (i % verbIter == 0)
        savePath!(Paths, XX, (i % saveIter == 0), skipForSave)
        ll, acc = impute!(ObsScheme(), Wnr, y, WWᵒ, WW, XXᵒ, XX, P, ll, fpt,
                          ρ=ρ, verbose=verbose, it=i)
        accImpCounter += 1*acc
        if paramUpdt
            imod = 1+i%updtLen
            recomputeODEs = any([e in dependsOnParams(P[1].Pt) for e
                                                       in idx(updtCoord[imod])])
            ll, acc, θ = updateParam!(ObsScheme(), updtType[imod], tKernel, θ,
                                      updtCoord[imod], y, WW, Pᵒ, P, XXᵒ, XX,
                                      ll, priors[imod], fpt, recomputeODEs;
                                      solver=ST(), verbose=verbose, it=i)
            accUpdtCounter[imod] += 1*acc
        end
        θchain[i+1] = copy(θ)
    end
    Time = collect(Iterators.flatten(p.tt[1:skipForSave:end-1] for p in P))
    θchain, accImpCounter/numSteps, accUpdtCounter./numSteps.*updtLen, Paths, Time
end
