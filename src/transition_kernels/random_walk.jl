using Distributions
import Random: rand!, rand
import Distributions: logpdf
import Base: eltype, length

"""
    RandomWalk(ϵ::T, pos::S)

Defines a random walk on `|ϵ|`-dimensional space. ``ϵ` defines the maximal
one-sided range of a single step and `pos` is a vector of indicators for whether
a respective index is restricted to take only positive values. For elements
restricted to take positive values, the update is done via: x⁽ⁿᵉʷ⁾ <- x⁽ᵒˡᵈ⁾eᵁ,
where U ∼ Unif(-ϵ,ϵ). For unrestricted: x⁽ⁿᵉʷ⁾ <- x⁽ᵒˡᵈ⁾ + U,
where U ∼ Unif(-ϵ,ϵ).
"""
struct RandomWalk{T,N}
    ϵ::NTuple{N,T}
    pos::NTuple{N,Bool}

    function RandomWalk(ϵ::S, pos=nothing
                        ) where S<:Union{T,Vector{T},NTuple{N,T}} where {N,T}
        _RandomWalk(T, ϵ, pos)
    end

    function _RandomWalk(::Type{T}, ϵ, pos::Nothing) where T
        ϵ, N = Tuple(ϵ), length(ϵ)
        new{T,N}(ϵ, Tuple([false for i in 1:N]))
    end

    function _RandomWalk(::Type{T}, ϵ, pos::Array{Any,1}) where T
        ϵ, N = Tuple(ϵ), length(ϵ)
        @assert length(pos) == 0
        new{T,N}(ϵ, Tuple([false for i in 1:N]))
    end

    function _RandomWalk(::Type{T}, ϵ, pos::R) where {T,R<:Union{S,Vector{S},NTuple{N1,S}}} where {N1,S<:Integer}
        ϵ, N = Tuple(ϵ), length(ϵ)
        @assert minimum(pos) >= 1 && maximum(pos) <= N
        new{T,N}(ϵ, Tuple([i in pos for i in 1:N]))
    end

    function _RandomWalk(::Type{T}, ϵ, pos::R) where {T,R<:Union{S,Vector{S},NTuple{N1,S}}} where {N1,S<:Bool}
        ϵ, N = Tuple(ϵ), length(ϵ)
        pos, N_temp = Tuple(pos), length(pos)
        @assert N == N_temp
        new{eltype(ϵ),N}(ϵ, pos)
    end

end

additiveStep = (x,pos)->pos ? 0.0 : x
multipStep = (x,pos)->pos ? exp(x) : 1.0
eltype(::RandomWalk{T}) where T = T
length(::RandomWalk{T,N}) where {T,N} = N

function new_tkernel(rw::RandomWalk, f, idx)
    ϵ = collect(rw.ϵ)
    ϵ[idx] = rw.pos[idx] ? exp(f(log(rw.ϵ[idx]))) : f(rw.ϵ[idx])
    RandomWalk(ϵ, rw.pos)
end

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
