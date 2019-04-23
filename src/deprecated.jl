"""
    ladderLength(::T, 𝓣Ladder, ladderOfPriors)

Retrieve total number of elements in a ladder
"""
ladderLength(::T, 𝓣Ladder, ladderOfPriors) where T = 1

function ladderLength(::T, 𝓣Ladder, ladderOfPriors
                      ) where T <: Union{SimulatedTemperingPriors,
                                         ParallelTemperingPriors}
    length(ladderOfPriors)
end

function ladderLength(::T, 𝓣Ladder, ladderOfPriors
                      ) where T <: Union{SimulatedTempering,ParallelTempering}
    length(𝓣Ladder)
end
"""
    pathLlikelihood!(θ, y, WW, Pᵒ, XXᵒ, 𝓣, ::ST=Ralston3())

Compute the log-likelihood of the imputed driving noise `WW`
"""

function pathLlikelihood!(θ, y, WW, Pᵒ, XXᵒ, 𝓣, ::ST=Ralston3()) where ST
    m = length(WW)
    for i in 1:m
        Pᵒ[i] = GuidPropBridge(Pᵒ[i], θ, 𝓣)
    end
    solveBackRec!(Pᵒ, ST())
    y₀ = copy(y)
    for i in 1:m
        solve!(Euler(), XXᵒ[i], y₀, WW[i], Pᵒ[i])
        y₀ = XXᵒ[i].yy[end]
    end
    llᵒ = 0.0
    for i in 1:m
        llᵒ += llikelihood(LeftRule(), XXᵒ[i], Pᵒ[i])
    end
    llᵒ
end


"""
    computeLogWeight(::T, ::ObsScheme, θ, y, WW, Pᵒ, XXᵒ, ll, ι,
                     ladderOfPriors, 𝓣Ladder, updtIdx, ::ST)

Compute log-weight for the element (θ, WW, 𝓣)
"""
function computeLogWeight(::T, ::ObsScheme, ::Any, ::Any, ::Any, ::Any,
                          ::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::ST
                          ) where {T,ObsScheme,ST}
    0.0
end

#NOTE parallel tempering must receive idx instead of ι
function computeLogWeight(::T, ::ObsScheme, θ, ::Any, ::Any, ::Any,
                          ::Any, ::Any, ι, ladderOfPriors, ::Any, updtIdx, ::ST
                          ) where {T <: Union{BiasingOfPriors,
                                              SimulatedTemperingPriors,
                                              ParallelTemperingPriors},
                                   ObsScheme,ST}
    ι == 1 && return 0.0
    logWeight = 0.0
    for (prior, priorᵒ) in zip(ladderOfPriors[1][updtIdx],
                               ladderOfPriors[ι][updtIdx])
        logWeight = logpdf(prior, θ) - logpdf(priorᵒ, θ)
    end
    logWeight
end

#NOTE Impossible to compute logWeights for FPT, parallel tempering must receive idx instead of ι
function computeLogWeight(::T, ::PartObs, θ, y, WW, Pᵒ, XXᵒ, ll, ι,
                          ::Any, 𝓣Ladder, ::Any, ::ST
                          ) where {T <: Union{SimulatedTempering,
                                              ParallelTempering}, ST}
    ι == 1 && return 0.0
    llᵒ = pathLlikelihood(θ, y, WW, Pᵒ, XXᵒ, 𝓣Ladder[ι], ST())
    llᵒ - ll
end
#- `cs`: parameters of the joint density, giving relative weights to ladder steps

"""
    updatePriorIdx!(::T, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any;
                         verbose=false, it=NaN)

By default no ladder
"""
function updatePriorIdx!(::T, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any;
                         verbose=false, it=NaN)
    1
end

"""
    updatePriorIdx!(::SimulatedTemperingPriors, ι, ladderOfPriors, cs, κ, θ,
                    accptMat, countMat; verbose=false, it=NaN)

Update position on the ladder of priors.
"""
function updatePriorIdx!(::SimulatedTemperingPriors, ι, ladderOfPriors, cs, κ,
                         θ, accptMat, countMat; verbose=false, it=NaN)
    ιᵒ = rand([max(ι-1, 1), min(ι+1, κ)])
    countMat[ι, ιᵒ] += 1
    if ιᵒ == ι
        llr = 0.0
    else
        llr = log(cs[ιᵒ]) - log(cs[ι])
        for (prior, priorᵒ) in zip(ladderOfPriors[ι], ladderOfPriors[ιᵒ])
            llr += logpdf(priorᵒ, θ) - logpdf(prior, θ)
        end
    end
    verbose && print("prior index update: ", it, " diff_ll: ",
                     round(llr, digits=3))
    if acceptSample(llr, verbose)
        accptMat[ι, ιᵒ] += 1
        return ιᵒ
    else
        return ι
    end
end

"""
    updatePriorIdx!(::SimulatedTemperingPriors, ι, ladderOfPriors, cs, κ, θ,
                    accptMat, countMat; verbose=false, it=NaN)

Update position on the ladder of priors.
"""
function updatePriorIdx!(::ParallelTemperingPriors, ι, ladderOfPriors, cs, κ,
                         θs, accptMat, countMat; verbose=false, it=NaN)
    idx = rand(1:length(ι)-1)
    ιᵒ = copy(ι)
    ιᵒ[idx], ιᵒ[idx+1] = ιᵒ[idx+1], ιᵒ[idx]

    countMat[ι, ιᵒ] += 1
    llr = 0.0
    for (prior, priorNext) in zip(ladderOfPriors[idx], ladderOfPriors[idx+1])
        llr += ( logpdf(prior, θs[ιᵒ[idx]]) + logpdf(priorNext, θs[ιᵒ[idx]+1])
                 - logpdf(prior, θs[ι[idx]]) + logpdf(priorNext, θs[ι[idx]+1]) )
    end

    verbose && print("prior index update: ", it, " diff_ll: ",
                     round(llr, digits=3))
    if acceptSample(llr, verbose)
        accptMat[ι, ιᵒ] += 1
        return ιᵒ
    else
        return ι
    end
end

function updateTemperature!(::T, ;solver=ST=Ralston3(), verbose=verbose, it=i) where ST
    ι, NaN
end


function updateTemperature!(::SimulatedTempering, 𝓣s, cs, κ, θ, y, WW, Pᵒ, P,
                            XXᵒ, XX, ll, priors, fpt, accptMat, countMat;
                            solver::ST = Ralston3(), verbose=verbose, it=i
                            ) where ST
    ιᵒ = rand([max(ι-1, 1), min(ι+1, κ)])
    countMat[ι, ιᵒ] += 1
    if ιᵒ == ι
        llr = 0.0
    else
        llᵒ = pathLlikelihood(θ, y, WW, Pᵒ, XXᵒ, 𝓣s[ιᵒ], ST())
        llr = llᵒ + log(cs[ιᵒ]) - ll - log(cs[ι])
    end
    verbose && print("prior index update: ", it, " diff_ll: ",
                     round(llr, digits=3))
    if acceptSample(llr, verbose)
        accptMat[ι, ιᵒ] += 1
        return ιᵒ, 𝓣s[ιᵒ]
    else
        return ι, 𝓣s[ι]
    end
end


function updateTemperature!(::ParallelTempering, 𝓣s, cs, κ, θ, y, WW, Pᵒ, P,
                            XXᵒ, XX, ll, priors, fpt, accptMat, countMat;
                            solver::ST = Ralston3(), verbose=verbose, it=i
                            ) where ST
    idx = rand(1:length(ι)-1)
    ιᵒ = copy(ι)
    ιᵒ[idx], ιᵒ[idx+1] = ιᵒ[idx+1], ιᵒ[idx]

    countMat[ι, ιᵒ] += 1
    llᵒ = (pathLlikelihood(θs[ιᵒ[idx]], y, WWs[ιᵒ[idx]], Pᵒs[idx],
                           XXᵒs[ιᵒ[idx]], 𝓣s[idx], ST())
           + pathLlikelihood(θs[ιᵒ[idx]+1], y, WWs[ιᵒ[idx]+1], Pᵒs[idx+1],
                             XXᵒs[ιᵒ[idx]+1], 𝓣s[idx+1], ST()))
    llr = llᵒ - lls[idx] - lls[idx+1]

    verbose && print("prior index update: ", it, " diff_ll: ",
                     round(llr, digits=3))
    if acceptSample(llr, verbose)
        accptMat[ι, ιᵒ] += 1
        return ιᵒ, 𝓣s[ιᵒ]
    else
        return ι, 𝓣s[ι]
    end
end


struct MCMCWorkspace
    m::Int64
    updtLen::Int64
    Wnr::TW
    WWᵒ::TWW
    WW::TWW
    XXᵒ::TXX
    XX::TXX
    Paths::Vector{Any}
    accImptCounter::Vector{Int64}
    accUpdtCounter::Vector{Int64}
    θchain::Vector{Tθ}
    recomputeODEs::Vector{Bool}

end
