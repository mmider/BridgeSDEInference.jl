using Bridge
using StaticArrays
import Bridge: b, σ, B, β, a, constdiff
const ℝ = SVector{N,T} where {N,T}
import Base.display
"""
    LotkaVolterraDiffusion <: ContinuousTimeProcess{ℝ{2}}
Struct defining a stochastic Lotka Volterra model
"""
struct LotkaVolterraDiffusion{T} <: ContinuousTimeProcess{ℝ{2,T}}
    α::T
    β::T
    γ::T
    δ::T
    σ1::T
    σ2::T
    function LotkaVolterraDiffusion(α::T, β, γ, δ, σ1, σ2) where T
        new{T}(α, β, γ, δ, σ1, σ2)
    end
end
function b(t, x, P::LotkaVolterraDiffusion{T}) where T
    ℝ{2}(P.α*x[1] - P.β*x[1]*x[2], P.δ*x[1]*x[2] - P.γ*x[2])
end
function σ(t, x, P::LotkaVolterraDiffusion{T}) where T
    SDiagonal(P.σ1, P.σ2)
end
constdiff(::LotkaVolterraDiffusion) = true
clone(P::LotkaVolterraDiffusion, θ) = LotkaVolterraDiffusion(θ...)
params(P::LotkaVolterraDiffusion) = [P.α, P.β, P.γ, P.δ, P.σ1, P.σ2]
domain(P::LotkaVolterraDiffusion) = LowerBoundedDomain((0.0, 0.0), (1,2))


# <---------------------------------------------
# this is optional, needed for conjugate updates

nonhypo(P::LotkaVolterraDiffusion, x) = x
@inline hypo_a_inv(P::LotkaVolterraDiffusion, t, x) = SDiagonal(1.0/P.σ1^2, 1.0/P.σ2^2)
num_non_hypo(P::Type{<:LotkaVolterraDiffusion}) = 2

phi(::Val{0}, t, x, P::LotkaVolterraDiffusion) = (zero(x[1]), zero(x[2]))
phi(::Val{1}, t, x, P::LotkaVolterraDiffusion) = (x[1], zero(x[2]))
phi(::Val{2}, t, x, P::LotkaVolterraDiffusion) = (-x[1]*x[2], zero(x[2]))
phi(::Val{3}, t, x, P::LotkaVolterraDiffusion) = (zero(x[1]), x[1]*x[2])
phi(::Val{4}, t, x, P::LotkaVolterraDiffusion) = (zero(x[1]), -x[2])
phi(::Val{5}, t, x, P::LotkaVolterraDiffusion) = (zero(x[1]), zero(x[2]))
phi(::Val{6}, t, x, P::LotkaVolterraDiffusion) = (zero(x[1]), zero(x[2]))

#
# <---------------------------------------------


struct LotkaVolterraDiffusionAux{R,S1,S2} <: ContinuousTimeProcess{ℝ{2,R}}
    α::R
    β::R
    γ::R
    δ::R
    σ1::R
    σ2::R
    t::Float64
    u::S1
    T::Float64
    v::S2

    function LotkaVolterraDiffusionAux(α::R, β, γ, δ, σ1, σ2, t::Float64, u::S1,
                                       T::Float64, v::S2) where {R,S1,S2}
        new{R,S1,S2}(α, β, γ, δ, σ1, σ2, t, u, T, v)
    end
end

function B(t, P::LotkaVolterraDiffusionAux{T,S1,S2}) where {T,S1,S2}
#    ℝ{2}(P.α*x[1] - P.β*x[1]*x[2], P.δ*x[1]*x[2] - P.γ*x[2])
    @SMatrix [-0.0 -P.β*P.γ/P.δ; P.α*P.δ/P.β -0.0]
end

# mean = ℝ{2}(P.γ/P.δ, P.α/P.β)
function β(t, P::LotkaVolterraDiffusionAux{T,S1,S2}) where {T,S1,S2}
    ℝ{2}(-P.γ/P.δ*P.α, P.α/P.β*P.γ)
end

function σ(t, P::LotkaVolterraDiffusionAux{T,S1,S2}) where {T,S1,S2}
    SDiagonal(P.σ1, P.σ2)
end

function σ(t, x, P::LotkaVolterraDiffusionAux{T,S1,S2}) where {T,S1,S2}
    SDiagonal(P.σ1, P.σ2)
end
depends_on_params(::LotkaVolterraDiffusionAux{T,S1,S2}) where {T,S1,S2} = (1, 2, 3, 4, 5, 6)

constdiff(::LotkaVolterraDiffusionAux{T,S1,S2}) where {T,S1,S2} = true
b(t, x, P::LotkaVolterraDiffusionAux{T,S1,S2}) where {T,S1,S2} = B(t,P) * x + β(t,P)
a(t, P::LotkaVolterraDiffusionAux{T,S1,S2}) where {T,S1,S2} = σ(t,P) * σ(t, P)'


clone(P::LotkaVolterraDiffusionAux, θ) = LotkaVolterraDiffusionAux(θ..., P.t,
                                                                   P.u, P.T, P.v)

clone(P::LotkaVolterraDiffusionAux, θ, v) = LotkaVolterraDiffusionAux(θ..., P.t,
                                                                      v, P.T, v)
params(P::LotkaVolterraDiffusionAux) = [P.α, P.β, P.γ, P.δ, P.σ1, P.σ2]
