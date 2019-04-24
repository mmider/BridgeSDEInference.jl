import Base: getindex, length

struct Priors
    priors
    indicesForUpdt::Array{Array{Int64,1},1}
    Priors(priors, indicesForUpdt) = new(priors, indicesForUpdt)
    Priors(priors) = new(priors, [[i] for i in 1:length(priors)])
end

getindex(p::Priors, i::Int) = p.priors[p.indicesForUpdt[i]]

function logpdf(p::Priors, θ)
    total = 0.0
    for prior in p.priors
        total += logpdf(prior, θ)
    end
    total
end

function logpdf(p::Priors, θ, updtIdx::Int)
    total = 0.0
    for prior in p[updtIdx]
        total += logpdf(prior, θ)
    end
    total
end


struct LadderOfPriors{N}
    ladder::NTuple{N, Priors}
    LadderOfPriors(ladder::NTuple{N, Priors}) where N = new{N}(ladder)
end

getindex(L::LadderOfPriors, i::Int) = L.ladder[i]
length(L::LadderOfPriors) = length(L.ladder)
