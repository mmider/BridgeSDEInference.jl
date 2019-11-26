import Base: getindex, length, iterate
import Distributions.logpdf

"""
    Priors

Struct
```
struct Priors
    priors
    indicesForUpdt::Array{Array{Int64,1},1}
end
```
holds relevant information about priors: distributions that need to be used
and the pattern in which MCMC makes use of those distributions. `priors`
is a list of distributions with all priors that are to be used. `indicesForUpdt`
is a list of lists---the reason is as follows. Markov chains can have multiple
transition kernels, each updating (possibly) multiple number of parameters
(think of Gibbs sampler, which updates parameters in blocks)---for each such
update, priors for all updated parameters are needed. The inner list of
`indicesForUpdt` gives indices of all priors that together (in a factories form)
make up a prior for a respective transition kernel of the Markov chain

    Priors(priors, indicesForUpdt)
Base constructor. `priors` is a list of priors, `indicesForUpdt` is a pattern
of indices indicating in which way parameters are being updated

    Priors(priors)
Convenience constructor. Most of the time each kernel of the Markov chain will
be updating only a single parameter and thus only a single prior will be needed
for each transition. In that case providing a list of priors in `priors` is
sufficient. This constructor takes care of the internal objects in such setting.
"""
struct Priors{S,T}
    priors::S
    coord_idx::T
    function Priors(priors, coord_idx)
        priors, coord_idx = (priors,), (coord_idx,)
        S, T = typeof(priors), typeof(coord_idx)
        new{S,T}(priors, coord_idx)
    end
    function Priors(priors::T1, coord_idx::T2) where {T1<:Tuple, T2<:Tuple}
        S, T = typeof(priors), typeof(coord_idx)
        new{S,T}(priors, coord_idx)
    end
end

"""
    getindex(p::Priors, i::Int)

Fetch the `i`-th list of priors (corresponding to the `i`-th transition kernel)
from the `p` object.
"""
getindex(p::Priors, i::Int) = p.priors[i]

"""
    logpdf(p::Priors, θ)
Compute the logarithm of a product of all priors in object `p`, evaluated at `θ`
"""
function logpdf(p::Priors, θ)
    total = 0.0
    for (prior,coords) in p
        total += logpdf(prior, coords, θ)
    end
    total
end

"""
    length(p::Priors)
Return the total number of priors held by the object `p`
"""
length(p::Priors) = length(p.priors)

"""
    iterate(iter::Priors, state=(1, 0))

Iterates over sets of priors defined for separate parameter update steps
"""
function iterate(iter::Priors, i=1)
    if i > length(iter)
        return nothing
    end
    return ((iter.priors[i], iter.coord_idx[i]), i + 1)
end


"""
    ExpUnif

Distribution of an exponential of a uniform random variable U,
where U ~ Unif([low, up])
"""
struct ExpUnif
    low::Float64 # lower bound on a log scale
    up::Float64 # upper bound on a log scale

    ExpUnif(low_bd, up_bd) = new(low_bd, up_bd)
end

function logpdf(p::ExpUnif, coord_idx, θ)
    θi = θ[[indices(coord_idx)...]]
    @assert length(θi) == 1
    lθi = log(θi[1])
    p.low <= lθi <= p.up ? -log(p.up-p.low) - lθi : -Inf
end
