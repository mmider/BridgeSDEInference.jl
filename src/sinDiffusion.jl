using Bridge
using Bridge.Models: ℝ
import Bridge: b, σ, B, β, a, constdiff

struct SinDiffusion <: ContinuousTimeProcess{ℝ{1}}
    a::Float64
    b::Float64
    c::Float64
    σ::Float64
end

b(t, x, P::SinDiffusion) = ℝ{1}(P.a + P.b*sin.(P.c * x))
σ(t, x, P::SinDiffusion) = ℝ{1}(P.σ)
constdiff(::SinDiffusion) = true
clone(::SinDiffusion, θ) = SinDiffusion(θ...)
clone(::SinDiffusion, θ, 𝓣) = clone(SinDiffusion(), (θ[1:3]..., 𝓣))
params(P::SinDiffusion) = [P.a, P.b, P.c, P.σ]

struct SinDiffusionAux <: ContinuousTimeProcess{ℝ{1}}
    a::Float64
    b::Float64
    c::Float64
    σ::Float64
    t::Float64
    u::Float64
    T::Float64
    v::Float64
end

B(t, P::SinDiffusionAux) = @SMatrix [0.0]
β(t, P::SinDiffusionAux) = ℝ{1}(0.0)
σ(t, P::SinDiffusionAux) = ℝ{1}(P.σ)
dependsOnParams(::SinDiffusionAux) = (4,)
constdiff(::SinDiffusionAux) = true
b(t, x, P::SinDiffusionAux) = B(t, P) * x + β(t, P)
a(t, P::SinDiffusionAux) = σ(t, P) * σ(t, P)'
clone(P::SinDiffusionAux, θ) = SinDiffusionAux(θ..., P.t, P.u, P.T, P.v)
clone(P::SinDiffusionAux, θ, 𝓣) = clone(P, (θ[1:3]..., 𝓣))
