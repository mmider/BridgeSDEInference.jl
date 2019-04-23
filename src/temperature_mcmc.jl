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
    initTemperature(::T, N, 𝓣Ladder, κ)

Initialise:
 - ι (current index on a ladder)
 - ιchain (history of ι)

...
# Arguments
- `::T`: type of MCMC sampler
- `N`: length with which ``ιchain` is to be initialised
- `𝓣Ladder`: temperature ladder
- `κ`: number of elements in a ladder
...
"""
function initTemperature(::VanillaMCMC, N, mcmcParams, ::Any, ::Any)
    ℒ = EmptyLadder()
    1, fill(1, N), ℒ
end

function initTemperature(::BiasingOfPriors, N, mcmcParams, ::Any, ::Any)
    @unpack priors, biasedPriors = mcmcParams
    ℒ = BiasedPr(Tuple(prior[1] for prior in priors[1]),
                 Tuple(prior[1] for prior in biasedPriors[1]))

    1, fill(1, N), ℒ
end

function ιForSimulated(N)
    ι = 1
    ιchain = Vector{Int64}(undef, N)
    ιchain[1] = ι
    ι, ιchain
end

function initTemperature(::SimulatedTemperingPriors, N, mcmcParams, ::Any, ::Any)
    @unpack ladderOfPriors = mcmcParams
    ι, ιchain = ιForSimulated(N)
    ℒ = SimTempPrLadder(ladderOfPriors, cs)
    ι, ιchain, ℒ
end

function initTemperature(::SimulatedTempering, N, mcmcParams, P, XX)
    @unpack cs, 𝓣Ladder = mcmcParams
    ι, ιchain = ιForSimulated(N)
    ℒ = SimTempLadder(𝓣Ladder, cs, P, XX)
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
    @unpack 𝓣Ladder = mcmcParams
    ι, ιchain = ιForParallel(𝓣Ladder, N)
    ℒ = ParTempLadder(𝓣Ladder, Ps, XXs)
    ι, ιchain, ℒ
end


function computeLogWeight!(ℒ::EmptyLadder, θ, y, WW, ι, ll, ::ST) where ST
    0.0
end

function computeLogWeight!(ℒ::BiasedPr, θ, y, WW, ι, ll, ::ST) where ST
    computeLogWeight!(ℒ, θ)
end

function computeLogWeight!(ℒ::SimTempPrLadder, θ, y, WW, ι, ll, ::ST) where ST
    computeLogWeight!(ℒ, θ, ι)
end

function computeLogWeight!(ℒ::SimTempLadder, θ, y, WW, ι, ll, ::ST) where ST
    computeLogWeight!(ℒ, θ, y, WW, ι, ll, ST())
end

function computeLogWeight!(ℒ::ParTempPrLadder, θ, y, WW, ι, idx, ll, ::ST) where ST
    computeLogWeight!(ℒ, θ, ι)
end

function computeLogWeight!(ℒ::ParTempLadder, θ, y, WW, ι, idx, ll, ::ST) where ST
    computeLogWeight!(ℒ, θ, y, WW, ι, idx, ll, ST())
end

function update!(ℒ::EmptyLadder, θ, y, WW, ι, ll, ::ST, verbose, it) where ST
    ι
end

function update!(ℒ::BiasedPr, θ, y, WW, ι, ll, ::ST, verbose, it) where ST
    ι
end

function update!(ℒ::SimTempPrLadder, θ, y, WW, ι, ll, ::ST, verbose, it) where ST
    update!(ℒ, θ, ι, ST(); verbose=vebose, it=it)
end

function update!(ℒ::SimTempLadder, θ, y, WW, ι, ll, ::ST, verbose, it) where ST
    update!(ℒ, θ, y, WW, ι, ll, ST(); verbose=vebose, it=it)
end

function update!(ℒ::ParTempPrLadder, θs, ys, WWs, ι, lls, ::ST, verbose, it) where ST
    update!(ℒ, θs, ι, ST(); verbose=vebose, it=it)
end

function update!(ℒ::ParTempLadder, θs, ys, WWs, ι, lls, ::ST, verbose, it) where ST
    udpate!(ℒ, θs, ys, WWs, ι, lls, ST(); verbose=vebose, it=it)
end

function formatChains(ℒ::T, ιchain, logω, saveIter) where T
    NaN, NaN
end

function formatChains(ℒ::T, ιchain, logω, saveIter) where T <: SimLadders
    M = length(logω)
    m = length(ιchain)
    𝓣chain = Vector{Tuple{Int64, Int64, Float64}}(undef, M)
    𝓣chainPth = Vector{Tuple{Int64, Int64, Float64}}(undef, div(m, saveIter))
    updtLen = div(M-1, m-1)

    𝓣chain[1] = (ιchain[1], get𝓣(ℒ, 𝓣Ladder, 1), logω[1])
    idx = 1
    pIdx = 1
    for i in 1:m
        if i % saveIter == 0
            𝓣chainPth[pIdx] = (ιchain[i], 𝓣Ladder(ℒ, ιchain[i]), logω[idx])
            pIdx += 1
        end
        for j in 1:updtLen
            idx += 1
            𝓣chain[idx] = (ιchain[i], 𝓣Ladder(ℒ, ιchain[i]), logω[idx])
        end
    end
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


function wmcmc(::MCMCType, ::ObsScheme, y, w, P˟, P̃, Ls, Σs,
               numSteps, tKernel, τ, mcmcParams; solver::ST=Ralston3()
               ) where {MCMCType, ObsScheme <: AbstractObsScheme, ST}
    (@unpack obs, obsTimes, fpt, ρ, dt, saveIter, verbIter, updtCoord,
             paramUpdt, skipForSave, updtType, biasedPriors = mcmcParams)
    P = findProposalLaw(obs, obsTimes, P˟, P̃, Ls, Σs, τ; dt=dt, solver=ST())
    m = length(obs)-1
    updtLen = length(updtCoord)
    Wnr, WWᵒ, WW, XXᵒ, XX, Pᵒ, ll = initialise(ObsScheme(), P, m, y, w, fpt)
    Paths = []
    numAccImp = 0
    numAccUpdt = [0 for i in 1:updtLen]
    θ = params(P˟)
    θchain = Vector{typeof(θ)}(undef, numSteps*updtLen+1)
    θchain[1] = copy(θ)
    recomputeODEs = [any([e in dependsOnParams(P[1].Pt) for e
                         in idx(uc)]) for uc in updtCoord]

    ι, ιchain, ℒ = initTemperature(MCMCType(), numSteps+1, mcmcParams, P, XX)
    logωs = Vector{Float64}(undef, numSteps*updtLen+1)
    logωs[1] = 0.0

    step = 1
    for i in 1:numSteps
        verbose = (i % verbIter == 0)
        savePath!(Paths, XX, (i % saveIter == 0), skipForSave)
        ll, acc = impute!(ObsScheme(), Wnr, y, WWᵒ, WW, XXᵒ, XX, P, ll, fpt,
                          ρ=ρ, verbose=verbose, it=i)
        numAccImp += 1*acc
        if paramUpdt
            for j in 1:updtLen
                ll, acc, θ = updateParam!(ObsScheme(), updtType[j], tKernel, θ,
                                          updtCoord[j], y, WW, Pᵒ, P, XXᵒ, XX,
                                          ll, biasedPriors[ι][j], fpt, recomputeODEs[j];
                                          solver=ST(), verbose=verbose, it=i)
                numAccUpdt[j] += 1*acc
                step += 1
                logωs[step] = computeLogWeight!(ℒ, θ, y, WW, ι, ll, ST())
                θchain[step] = copy(θ)
            end
            verbose && print("------------------------------------------------",
                             "------\n")
        end
        ι = update!(ℒ, θ, y, WW, ι, ll, ST(), verbose, i)
        ιchain[i+1] = ι
    end
    Time = collect(Iterators.flatten(p.tt[1:skipForSave:end-1] for p in P))
    𝓣chain, 𝓣chainPth = formatChains(ℒ, ιchain, logωs, saveIter)
    (θchain, 𝓣chain, logωs, numAccImp/numSteps, numAccUpdt./numSteps, accptRate(ℒ),
     Paths, 𝓣chainPth, Time)
end
