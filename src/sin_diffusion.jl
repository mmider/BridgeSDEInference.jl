using Bridge
using Bridge.Models: ℝ
import Bridge: b, σ, B, β, a, constdiff

struct SinDiffusion{T} <: ContinuousTimeProcess{ℝ{1, T}}
    a::T
    b::T
    c::T
    σ::T
end

b(t, x, P::SinDiffusion) = ℝ{1}(P.a + P.b*sin.(P.c * x))
σ(t, x, P::SinDiffusion) = ℝ{1}(P.σ)
constdiff(::SinDiffusion) = true
clone(::SinDiffusion, θ) = SinDiffusion(θ...)
params(P::SinDiffusion) = [P.a, P.b, P.c, P.σ]

struct SinDiffusionAux{T,S} <: ContinuousTimeProcess{ℝ{1, T}}
    a::T
    b::T
    c::T
    σ::T
    t::Float64
    u::S
    T::Float64
    v::S
end

B(t, P::SinDiffusionAux) = @SMatrix [0.0]
β(t, P::SinDiffusionAux) = ℝ{1}(0.0)
σ(t, P::SinDiffusionAux) = ℝ{1}(P.σ)
dependsOnParams(::SinDiffusionAux) = (4,)
constdiff(::SinDiffusionAux) = true
b(t, x, P::SinDiffusionAux) = B(t, P) * x + β(t, P)
a(t, P::SinDiffusionAux) = σ(t, P) * σ(t, P)'
clone(P::SinDiffusionAux, θ) = SinDiffusionAux(θ..., P.t, P.u, P.T, P.v)
