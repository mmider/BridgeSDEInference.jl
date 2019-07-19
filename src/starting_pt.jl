using GaussianDistributions, Random
import Random.rand
import Distributions.logpdf

"""
    StartingPtPrior

Types inheriting from the abstract type `StartingPtPrior` indicate the prior
that is put on the starting point of the observed path following the dynamics
of some stochastic differential equation.
"""
abstract type StartingPtPrior end

"""
    KnownStartingPt{T} <: StartingPtPrior

Indicates that the starting point is known and stores its value in `y`
"""
struct KnownStartingPt{T} <: StartingPtPrior where T
    y::T
    KnownStartingPt(y::T) where T = new{T}(y)
end

"""
    GsnStartingPt{T,S} <: StartingPtPrior

Indicates that the starting point is equipped with a Gaussian prior with
mean `μ` and covariance matrix `Σ`. It also stores the most recently sampled
starting point `y`

    GsnStartingPt(y::T, μ::T, Σ::S)

Base constructor explicitly initialising all parameters

    GsnStartingPt(μ::T, Σ::S)

Base constructor v2. It initialises the mean `μ` and covariance `Σ` parameters
and samples the starting point `y` according to a prior

    GsnStartingPt(y::T, G::GsnStartingPt{T,S})

Copy constructor. It copies the mean `μ` and covariance `Σ` from the old prior
`G` and sets a new sampled starting point `y`
"""
struct GsnStartingPt{T,S} <: StartingPtPrior where {T,S}
    y::T
    μ::T
    Σ::S

    function GsnStartingPt(y::T, μ::T, Σ::S) where {T,S}
        new{T,S}(y, μ, Σ)
    end

    function GsnStartingPt(μ::T, Σ::S) where {T,S}
        y = rand(Gaussian(μ, Σ))
        new{T,S}(y, μ, Σ)
    end

    function GsnStartingPt(y::T, G::GsnStartingPt{T,S}) where {T,S}
        new{T,S}(y, G.μ, G.Σ)
    end
end

"""
    rand(G::GsnStartingPt)

Sample a new starting point according to the prior distribution
"""
rand(G::GsnStartingPt) = GsnStartingPt(rand(Gaussian(G.μ, G.Σ)), G)

"""
    rand(G::KnownStartingPt)

Sample a new starting point according to the prior distribution
"""
rand(G::KnownStartingPt) = G #TODO make sure that `copy(G)` is not needed
