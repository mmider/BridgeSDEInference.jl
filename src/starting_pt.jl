using GaussianDistributions, Random
import Random.rand
import Distributions.logpdf
import GaussianDistributions: whiten, unwhiten

"""
    StartingPtPrior

Types inheriting from the abstract type `StartingPtPrior` indicate the prior
that is put on the starting point of the observed path following the dynamics
of some stochastic differential equation.
"""
abstract type StartingPtPrior{T} end

"""
    KnownStartingPt{T} <: StartingPtPrior

Indicates that the starting point is known and stores its value in `y`
"""
struct KnownStartingPt{T} <: StartingPtPrior{T}
    y::T
    KnownStartingPt(y::T) where T = new{T}(y)
end

"""
    GsnStartingPt{T,S} <: StartingPtPrior

Indicates that the starting point is equipped with a Gaussian prior with
mean `μ` and covariance matrix `Σ`. It also stores the most recently sampled
white noise `z` used to compute the starting point and a precision matrix
`Λ`:=`Σ`⁻¹. `μ₀` and `Σ₀` are the mean and covariance of the white noise

    GsnStartingPt(y::T, μ::T, Σ::S)

Base constructor explicitly initialising starting point `y` (on top of mean `μ`
and covariance matrix `Σ`) which is internally transformed to its corresponding
white noise `z`

    GsnStartingPt(μ::T, Σ::S)

Base constructor v2. It initialises the mean `μ` and covariance `Σ` parameters
and samples white noise `z`. `Λ` is set according to `Λ`:=`Σ`⁻¹

    GsnStartingPt(z::T, G::GsnStartingPt{T,S})

Copy constructor. It copies the mean `μ`, covariance `Σ`, precision `Λ` as well
as mean of the white noise `μ₀` and covariance of the white noise `Σ₀` from
the old prior `G` and sets a new white noise to `z`.
"""
struct GsnStartingPt{T,S} <: StartingPtPrior{T} where {S}
    z::T
    μ::T
    Σ::S
    Λ::S
    μ₀::T
    Σ₀::UniformScaling

    function GsnStartingPt(y::T, μ::T, Σ::S) where {T,S}
        z = whiten(Σ, y-μ)
        new{T,S}(z, μ, Σ, inv(Σ), 0*μ, I)
    end

    function GsnStartingPt(μ::T, Σ::S) where {T,S}
        μ₀ = 0*μ
        Σ₀ = I
        z = rand(Gaussian(μ₀, Σ₀))
        new{T,S}(z, μ, Σ, inv(Σ), μ₀, Σ₀)
    end

    function GsnStartingPt(z::T, G::GsnStartingPt{T,S}) where {T,S}
        new{T,S}(z, G.μ, G.Σ, G.Λ, G.μ₀, G.Σ₀)
    end
end


"""
    rand(G::GsnStartingPt, ρ)

Sample new white noise using Crank-Nicolson scheme with memory parameter `ρ` and
a previous value of the white noise stored inside object `G`
"""
function rand(G::GsnStartingPt, ρ=0.0)
    zᵒ = rand(Gaussian(G.μ₀, G.Σ₀))
    z = √(1-ρ)*zᵒ + √(ρ)*G.z # preconditioned Crank-Nicolson
    GsnStartingPt(z, G)
end

"""
    rand(G::KnownStartingPt, ::Any)

If starting point is known then nothing can be sampled
"""
rand(G::KnownStartingPt, ::Any=nothing) = G


"""
    startPt(G::GsnStartingPt, P)

Compute a new starting point from the white noise for a given posterior
distribution obtained from combining prior `G` and the likelihood encoded by the
object `P`.
"""
function startPt(G::GsnStartingPt, P::GuidPropBridge)
    μₚₒₛₜ = (P.H[1] + G.Λ) \ (P.Hν[1] + G.Λ * G.μ)
    Σₚₒₛₜ = inv(P.H[1] + G.Λ)
    Σₚₒₛₜ = 0.5 * (Σₚₒₛₜ + Σₚₒₛₜ') # remove numerical inaccuracies
    unwhiten(Σₚₒₛₜ, G.z) + μₚₒₛₜ
end


"""
    startPt(G::GsnStartingPt, P)

Compute a new starting point from the white noise for a given prior
distribution `G`
"""
startPt(G::GsnStartingPt) = unwhiten(G.Σ, G.z) + G.μ

"""
    startPt(G::KnownStartingPt, P)

Return starting point
"""
startPt(G::KnownStartingPt, P::GuidPropBridge) = G.y

"""
    startPt(G::KnownStartingPt, P)

Return starting point
"""
startPt(G::KnownStartingPt) = G.y

"""
    invStartPt(y, G::GsnStartingPt, P::GuidPropBridge)

Compute the driving noise that is needed to obtain starting point `y` under
prior `G` and likelihood `P`. Return a new starting point object
"""
function invStartPt(y, G::GsnStartingPt, P::GuidPropBridge)
    μₚₒₛₜ = (P.H[1] + G.Λ) \ (P.Hν[1] + G.Λ * G.μ)
    Σₚₒₛₜ = inv(P.H[1] + G.Λ)
    Σₚₒₛₜ = 0.5 * (Σₚₒₛₜ + Σₚₒₛₜ')
    GsnStartingPt(whiten(Σₚₒₛₜ, y-μₚₒₛₜ), G)
end

"""
    invStartPt(y, G::KnownStartingPt, P::GuidPropBridge)

Starting point known, no need for dealing with white noise
"""
invStartPt(y, G::KnownStartingPt, P::GuidPropBridge) = G

"""
    logpdf(::KnownStartingPt, y)

Nothing to do so long as `y` is equal to the known starting point inside `G`
"""
function logpdf(G::KnownStartingPt, y)
    (G.y == y) ? 0.0 : error("Starting point is known, but a different value ",
                             "was passed to logpdf.")
end


"""
    logpdf(::GsnStartingPt, y)

log-probability density function evaluated at `y` of a prior distribution `G`
"""
logpdf(G::GsnStartingPt, y) = logpdf(Gaussian(G.μ, G.Σ), y)
