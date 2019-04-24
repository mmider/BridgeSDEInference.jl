using Parameters

"""
    MCMCSampler

Types inheriting from abstract type `MCMCSampler` define the type of Markov
chain Monte Carlo algorithm to use
"""
abstract type MCMCSampler end

"""
    VanillaMCMC <: MCMCSampler

Regular MCMC sampler
"""
struct VanillaMCMC <: MCMCSampler end

"""
    BiasingOfPriors <: MCMCSampler

Regular MCMC sampler with priors substituted for other, biased priors. The bias
is corrected with an importance sampling step, which results in a weighted chain
"""
struct BiasingOfPriors <: MCMCSampler end

"""
    SimulatedTemperingPriors <: MCMCSampler

Simulated Tempering algorithm, which instead of a ladder of temperatures uses
a ladder of priors, with a gradient of biasing strengths
"""
struct SimulatedTemperingPriors <: MCMCSampler end

"""
    SimulatedTempering <: MCMCSampler

Simulated Tempering algorithm
"""
struct SimulatedTempering <: MCMCSampler end

"""
    ParallelTemperingPriors <: MCMCSampler

Parallel Tempering algorithm, which instead of a ladder of temperatures uses a
ladder of priors, with a gradient of biasing strengths
"""
struct ParallelTemperingPriors <: MCMCSampler end

"""
    ParallelTempering <: MCMCSampler

Parallel Tempering algorithm
"""
struct ParallelTempering <: MCMCSampler end


"""
    initTemperature(::VanillaMCMC, N, mcmcParams, ::Any, ::Any)

Set a ladder variable ℒ to a placeholder with information about a prior
"""
function initTemperature(::VanillaMCMC, N, mcmcParams, ::Any, ::Any)
    @unpack priors = mcmcParams
    ℒ = EmptyLadder(priors)
    1, fill(1, N), ℒ
end

"""
    initTemperature(::BiasingOfPriors, N, mcmcParams, ::Any, ::Any)

Set a ladder variable ℒ to a placeholder with information about a prior and a
biased prior
"""
function initTemperature(::BiasingOfPriors, N, mcmcParams, ::Any, ::Any)
    @unpack priors, biasedPriors = mcmcParams
    ℒ = BiasedPr(priors, biasedPriors)

    1, fill(1, N), ℒ
end

"""
    ιForSimulated(N)

Set ι (index on a ladder) and ιchain (history of ι's)
"""
function ιForSimulated(N)
    ι = 1
    ιchain = Vector{Int64}(undef, N)
    ιchain[1] = ι
    ι, ιchain
end


"""
    initTemperature(::SimulatedTemperingPriors, N, mcmcParams, ::Any, ::Any)

Set a ladder of priors ℒ, starting ι (index on a ladder) and ιchain (history of
ι's)
"""
function initTemperature(::SimulatedTemperingPriors, N, mcmcParams, ::Any, ::Any)
    @unpack ladderOfPriors = mcmcParams
    ι, ιchain = ιForSimulated(N)
    ℒ = SimTempPrLadder(ladderOfPriors, cs)
    ι, ιchain, ℒ
end

function initTemperature(::SimulatedTempering, N, mcmcParams, P, XX)
    @unpack priors, cs, 𝓣Ladder = mcmcParams
    ι, ιchain = ιForSimulated(N)
    ℒ = SimTempLadder(𝓣Ladder, cs, P, XX, priors)
    ι, ιchain, ℒ
end

function ιForParallel(ladder, N)
    ι = collect(1:length(ladder))
    ιchain = Vector{typeof(ι)}(undef, N)
    ιchain[1] .= ι
    ι, ιchain
end

function initTemperature(::ParallelTemperingPriors, N, mcmcParams, ::Any, ::Any)
    @unpack ladderOfPriors = mcmcParams
    ι, ιchain = ιForParallel(ladderOfPriors, N)
    ℒ = ParTempPrLadder(ladderOfPriors)
    ι, ιchain, ℒ
end

function initTemperature(::ParallelTempering, N, mcmcParams, Ps, XXs)
    @unpack priors, 𝓣Ladder = mcmcParams
    ι, ιchain = ιForParallel(𝓣Ladder, N)
    ℒ = ParTempLadder(𝓣Ladder, Ps, XXs, priors)
    ι, ιchain, ℒ
end

function createθchain(::T, θ, numSteps, updtLen
                      ) where T <: Union{VanillaMCMC,BiasingOfPriors}
    Vector{typeof(θ)}(undef, numSteps*updtLen+1)
end

function createθchain(::T, θ, numSteps, updtLen
                      ) where T <: Union{SimulatedTemperingPriors,
                                         SimulatedTempering,
                                         ParallelTemperingPriors,
                                         ParallelTempering}
    Vector{typeof(θ)}(undef, numSteps*(updtLen+1)+1)
end

"""
    computeLogWeight!(ℒ::EmptyLadder, θ, y, WW, ι, ll, ::ST)

Find a logarithm of weight for a sample (θ, y, WW, ι)
"""
function computeLogWeight!(ℒ::EmptyLadder, θ, y, WW, ι, ll, ::ST) where ST
    0.0
end

"""
    computeLogWeight!(ℒ::BiasedPr, θ, y, WW, ι, ll, ::ST)

Find a logarithm of weight for a sample (θ, y, WW, ι)
"""
function computeLogWeight!(ℒ::BiasedPr, θ, y, WW, ι, ll, ::ST) where ST
    computeLogWeight!(ℒ, θ)
end

"""
    computeLogWeight!(ℒ::SimTempPrLadder, θ, y, WW, ι, ll, ::ST)

Find a logarithm of weight for a sample (θ, y, WW, ι)
"""
function computeLogWeight!(ℒ::SimTempPrLadder, θ, y, WW, ι, ll, ::ST) where ST
    computeLogWeight(ℒ, θ, ι)
end

function computeLogWeight!(ℒ::SimTempLadder, θ, y, WW, ι, ll, ::ST) where ST
    computeLogWeight!(ℒ, θ, y, WW, ι, ll, ST())
end

function computeLogWeight!(ℒ::ParTempPrLadder, θ, y, WW, ι, idx, ll, ::ST) where ST
    computeLogWeight(ℒ, θ, ι)
end

function computeLogWeight!(ℒ::ParTempLadder, θ, y, WW, ι, idx, ll, ::ST) where ST
    computeLogWeight!(ℒ, θ, y, WW, ι, idx, ll, ST())
end

"""
    update!(ℒ::EmptyLadder, θ, y, WW, ι, ll, ::ST, verbose, it)

No ladder, no need to update anything
"""
function update!(ℒ::EmptyLadder, θ, y, WW, ι, ll, ::ST, verbose, it) where ST
    ι
end

"""
    update!(ℒ::BiasedPr, θ, y, WW, ι, ll, ::ST, verbose, it)

No ladder, no need to update anything
"""
function update!(ℒ::BiasedPr, θ, y, WW, ι, ll, ::ST, verbose, it) where ST
    ι
end

"""
    update!(ℒ::SimTempPrLadder, θ, y, WW, ι, ll, ::ST, verbose, it)

Update ι---a position on a ladder
"""
function update!(ℒ::SimTempPrLadder, θ, y, WW, ι, ll, ::ST, verbose, it) where ST
    update!(ℒ, θ, ι, ST(); verbose=verbose, it=it)
end

function update!(ℒ::SimTempLadder, θ, y, WW, ι, ll, ::ST, verbose, it) where ST
    update!(ℒ, θ, y, WW, ι, ll, ST(); verbose=verbose, it=it)
end

function update!(ℒ::ParTempPrLadder, θs, ys, WWs, ι, lls, ::ST, verbose, it) where ST
    update!(ℒ, θs, ι, ST(); verbose=verbose, it=it)
end

function update!(ℒ::ParTempLadder, θs, ys, WWs, ι, lls, ::ST, verbose, it) where ST
    udpate!(ℒ, θs, ys, WWs, ι, lls, ST(); verbose=verbose, it=it)
end

"""
    formatChains(ℒ::T, ιchain, logω, savedAtIdx)

No ladder, no need to return ladder positions and temperatures
"""
function formatChains(ℒ::T, ιchain, logω, savedAtIdx) where T
    NaN, NaN
end

"""
    formatChains(ℒ::T, ιchain, logω, savedAtIdx)

Format a chain with history of (ι, 𝓣, logω) corresponding to samples on a θchain
(ι, 𝓣, logω) are respectively position on a ladder, temperature level and
log-weight. Do the same for the saved paths.
"""
function formatChains(ℒ::T, ιchain, logωs, savedAtIdx) where T <: SimLadders
    𝓣chain = [(ι, 𝓣ladder(ℒ, ι), logω) for (ι, logω) in zip(ιchain, logωs)]
    𝓣chainPth = 𝓣chain[savedAtIdx]
    𝓣chain, 𝓣chainPth
end

@with_kw struct MCMCParams
    obs
    obsTimes
    priors
    fpt = fill(NaN, length(obsTimes)-1)
    ρ = 0.0
    dt = 1/5000
    saveIter = NaN
    verbIter = NaN
    updtCoord = (Val((true,)),)
    paramUpdt = true
    skipForSave = 1
    updtType = (MetropolisHastingsUpdt(),)
    cs = NaN
    biasedPriors = priors
    ladderOfPriors = NaN
    𝓣Ladder = NaN
end

function updateι(::T) where T
    true
end

function updateι(::T) where T <: Union{VanillaMCMC, BiasingOfPriors}
    false
end

function wmcmc(::MCMCType, ::ObsScheme, y, w, P˟, P̃, Ls, Σs,
               numSteps, tKernel, τ, mcmcParams; solver::ST=Ralston3()
               ) where {MCMCType, ObsScheme <: AbstractObsScheme, ST}
    (@unpack obs, obsTimes, fpt, ρ, dt, saveIter, verbIter, updtCoord,
             paramUpdt, skipForSave, updtType = mcmcParams)
    P = findProposalLaw(obs, obsTimes, P˟, P̃, Ls, Σs, τ; dt=dt, solver=ST())
    m = length(obs)-1
    updtLen = length(updtCoord)
    Wnr, WWᵒ, WW, XXᵒ, XX, Pᵒ, ll = initialise(ObsScheme(), P, m, y, w, fpt)
    Paths = []
    numAccImp = 0
    numAccUpdt = [0 for i in 1:updtLen]
    θ = params(P˟)
    θchain = createθchain(MCMCType(), θ, numSteps, updtLen)
    θchain[1] = copy(θ)
    recomputeODEs = [any([e in dependsOnParams(P[1].Pt) for e
                         in idx(uc)]) for uc in updtCoord]

    ι, ιchain, ℒ = initTemperature(MCMCType(), length(θchain), mcmcParams, P, XX)
    logωs = Vector{Float64}(undef, length(θchain))
    logωs[1] = 0.0

    step = 1
    savedAtIdx = []
    for i in 1:numSteps
        verbose = (i % verbIter == 0)
        savePath!(Paths, XX, (i % saveIter == 0), skipForSave)
        if (i % saveIter == 0)
            push!(savedAtIdx, step)
        end
        ll, acc = impute!(ObsScheme(), Wnr, y, WWᵒ, WW, XXᵒ, XX, P, ll, fpt,
                          ρ=ρ, verbose=verbose, it=i)
        numAccImp += 1*acc
        if paramUpdt
            for j in 1:updtLen
                ll, acc, θ = updateParam!(ObsScheme(), updtType[j], tKernel, θ,
                                          updtCoord[j], y, WW, Pᵒ, P, XXᵒ, XX,
                                          ll, prior(ℒ,ι,j), fpt, recomputeODEs[j];
                                          solver=ST(), verbose=verbose, it=i)
                numAccUpdt[j] += 1*acc
                step += 1
                logωs[step] = computeLogWeight!(ℒ, θ, y, WW, ι, ll, ST())
                θchain[step] = copy(θ)
                ιchain[step] = ι
            end
            if updateι(MCMCType())
                ι = update!(ℒ, θ, y, WW, ι, ll, ST(), verbose, i)
                step += 1
                logωs[step] = computeLogWeight!(ℒ, θ, y, WW, ι, ll, ST())
                θchain[step] = copy(θ)
                ιchain[step] = ι
            end

            verbose && print("------------------------------------------------",
                             "------\n")
        end
    end
    Time = collect(Iterators.flatten(p.tt[1:skipForSave:end-1] for p in P))
    𝓣chain, 𝓣chainPth = formatChains(ℒ, ιchain, logωs, savedAtIdx)
    (θchain, 𝓣chain, logωs, numAccImp/numSteps, numAccUpdt./numSteps, accptRate(ℒ),
     Paths, 𝓣chainPth, Time)
end
