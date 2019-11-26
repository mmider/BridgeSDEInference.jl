using Bridge
using StaticArrays, LinearAlgebra
import Bridge: b, σ, B, β, a, constdiff
const ℝ = SVector{N,T} where {N,T}
import Base.resize!


struct LorenzNS{T,N} <: ContinuousTimeProcess{ℝ{N,T}}
    θ::T
    σ::T
    function LorenzNS(θ::T, σ::T, N) where T
        new{T,N}(θ, σ)
    end
end

function b(t, x, P::LorenzNS{T,N}) where {T,N}
    out = copy(x)
    for i in 1:N
        out[i] = ( (x[mod1(i+1, N)]-x[mod1(i-2, N)])*x[mod1(i-1, N)]
                  - x[mod1(i, N)] + P.θ )
    end
    out
end

function σ(t, x, P::LorenzNS{T,N}) where {T,N}
    P.σ * Matrix{Float64}(I, N, N)
end

constdiff(::LorenzNS) = true
clone(P::LorenzNS, θ) = LorenzNS(θ...)
params(P::LorenzNS) = [P.θ, P.σ]

nonhypo(P::LorenzNS, x) = x
@inline hypo_a_inv(P::LorenzNS, t, x) = Matrix{Float64}(I, N, N)/P.σ^2
num_non_hypo(P::Type{<:LorenzNS{T,N}}) where {T,N} = N




P = LorenzNS(1.0, 2.0, 20)
b(0.0, [i for i in 1:20], P)
