using Distributions
import Random: rand!, rand
import Distributions: logpdf

"""
    RandomWalk(θ::T, ϵ::T, pos::S)

Defines a random walk on `|θ|`-dimensional space. `θ` is the initial state,
`ϵ` defines the maximal one-sided range of a single step and `pos` is a vector
of indicators for whether a respective index is restricted to take only positive
values. For elements restricted to take positive values, the update is done via:
x⁽ⁿᵉʷ⁾ <- x⁽ᵒˡᵈ⁾eᵁ, where U ∼ Unif(-ϵ,ϵ). For unrestricted:
x⁽ⁿᵉʷ⁾ <- x⁽ᵒˡᵈ⁾ + U, where U ∼ Unif(-ϵ,ϵ).
"""
mutable struct RandomWalk{T, S}
    θ::T
    ϵ::T
    pos::S
    RandomWalk(θ::T, ϵ::T, pos::S) where {T,S} = new{T,S}(θ,ϵ,pos)
    RandomWalk(rw::RandomWalk{T,S}, θ) where {T,S} = new{T,S}(θ, rw.ϵ, rw.pos)
end

additiveStep = (x,pos)->pos ? 0.0 : x
multipStep = (x,pos)->pos ? exp(x) : 1.0

"""
    rand!(rw::RandomWalk)

Update all elements of a random walker in-place.
"""
function rand!(rw::RandomWalk)
    rw.θ .+= additiveStep.(rand.(map(Uniform,-rw.ϵ, rw.ϵ)), rw.pos)
    rw.θ .*= multipStep.(rand.(map(Uniform,-rw.ϵ, rw.ϵ)), rw.pos)
    rw.θ
end

"""
    rand(rw::RandomWalk)

Return a newly sampled state of a random walker, with all element updated.
"""
function rand(rw::RandomWalk)
    θ = copy(rw.θ)
    θ .+= additiveStep.(rand.(map(Uniform,-rw.ϵ, rw.ϵ)), rw.pos)
    θ .*= multipStep.(rand.(map(Uniform,-rw.ϵ, rw.ϵ)), rw.pos)
    θ
end

"""
    rand!(rw::RandomWalk, ::UpdtIdx)

Update elements of a random walker on indices specified by the object `UpdtIdx`
in-place.
"""
function rand!(rw::RandomWalk, ::UpdtIdx) where UpdtIdx
    for i in idx(UpdtIdx())
        rw.θ[i] += additiveStep(rand(Uniform(-rw.ϵ[i], rw.ϵ[i])), rw.pos[i])
        rw.θ[i] *= multipStep(rand(Uniform(-rw.ϵ[i], rw.ϵ[i])), rw.pos[i])
    end
    rw.θ
end

"""
    rand(rw::RandomWalk, ::UpdtIdx)

Return a newly sampled state of a random walker, with updated elements only on
the indices specified by the object `UpdtIdx`.
"""
function rand(rw::RandomWalk, ::UpdtIdx) where UpdtIdx
    θ = copy(rw.θ)
    for i in idx(UpdtIdx())
        θ[i] += additiveStep(rand(Uniform(-rw.ϵ[i], rw.ϵ[i])), rw.pos[i])
        θ[i] *= multipStep(rand(Uniform(-rw.ϵ[i], rw.ϵ[i])), rw.pos[i])
    end
    θ
end

"""
    logpdf(rw::RandomWalk,θ,θᵒ)
Log-transition density of a random walker `rw` for going from `θ` to `θᵒ`.
"""
logpdf(rw::RandomWalk,θ,θᵒ) = sum( map((x,ϵ,pos)->pos ? -log(2.0*ϵ)-log(x) : 0.0,
                                       θᵒ, rw.ϵ, rw.pos) )
