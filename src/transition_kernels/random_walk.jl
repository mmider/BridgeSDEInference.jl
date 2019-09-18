using Distributions
import Random: rand!, rand
import Distributions: logpdf

"""
    RandomWalk(ϵ::T, pos::S)

Defines a random walk on `|ϵ|`-dimensional space. ``ϵ` defines the maximal
one-sided range of a single step and `pos` is a vector of indicators for whether
a respective index is restricted to take only positive values. For elements
restricted to take positive values, the update is done via: x⁽ⁿᵉʷ⁾ <- x⁽ᵒˡᵈ⁾eᵁ,
where U ∼ Unif(-ϵ,ϵ). For unrestricted: x⁽ⁿᵉʷ⁾ <- x⁽ᵒˡᵈ⁾ + U,
where U ∼ Unif(-ϵ,ϵ).
"""
struct RandomWalk{T, S}
    ϵ::T
    pos::S
    RandomWalk(ϵ::T, pos::S) where {T,S} = new{T,S}(ϵ,pos)
end

additiveStep = (x,pos)->pos ? 0.0 : x
multipStep = (x,pos)->pos ? exp(x) : 1.0

"""
    rand!(rw::RandomWalk, θ)

Update all elements of a random walker in-place.
"""
function rand!(rw::RandomWalk, θ)
    θ .+= additiveStep.(rand.(map(Uniform,-rw.ϵ, rw.ϵ)), rw.pos)
    θ .*= multipStep.(rand.(map(Uniform,-rw.ϵ, rw.ϵ)), rw.pos)
    θ
end

"""
    rand(rw::RandomWalk, θ)

Return a newly sampled state of a random walker, with all element updated.
"""
function rand(rw::RandomWalk, θ)
    θc = copy(θ)
    rand!(rw, θc)
end

"""
    rand!(rw::RandomWalk, θ, ::UpdtIdx)

Update elements of a random walker on indices specified by the object `UpdtIdx`
in-place.
"""
function rand!(rw::RandomWalk, θ, ::UpdtIdx) where UpdtIdx
    for i in idx(UpdtIdx())
        θ[i] += additiveStep(rand(Uniform(-rw.ϵ[i], rw.ϵ[i])), rw.pos[i])
        θ[i] *= multipStep(rand(Uniform(-rw.ϵ[i], rw.ϵ[i])), rw.pos[i])
    end
    θ
end

"""
    rand(rw::RandomWalk, θ, ::UpdtIdx)

Return a newly sampled state of a random walker, with updated elements only on
the indices specified by the object `UpdtIdx`.
"""
function rand(rw::RandomWalk, θ, ::UpdtIdx) where UpdtIdx
    θc = copy(θ)
    rand!(rw, θc, UpdtIdx())
end




"""
    logpdf(rw::RandomWalk, θ, θᵒ)
Log-transition density of a random walker `rw` for going from `θ` to `θᵒ`.
"""
logpdf(rw::RandomWalk, θ, θᵒ) = sum( map((x,ϵ,pos)->pos ? -log(2.0*ϵ)-log(x) :
                                                        0.0, θᵒ, rw.ϵ, rw.pos) )
