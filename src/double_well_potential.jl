using Bridge
using Bridge.Models: ℝ
import Bridge: b, σ, B, β, a, constdiff

struct DoubleWellPotential <: ContinuousTimeProcess{ℝ{1}}
    ρ::Float64
    μ::Float64
    σ::Float64
end

b(t, x, P::DoubleWellPotential) = ℝ{1}(-P.ρ*x[1]*(x[1]^2-P.μ))
σ(t, x, P::DoubleWellPotential) = ℝ{1}(P.σ)
constdiff(::DoubleWellPotential) = true
clone(::DoubleWellPotential, θ) = DoubleWellPotential(θ...)
clone(::DoubleWellPotential, θ, 𝓣) = clone(DoubleWellPotential(), (θ[1:2]..., 𝓣))
params(P::DoubleWellPotential) = [P.ρ, P.μ, P.σ]

struct DoubleWellPotentialAux <: ContinuousTimeProcess{ℝ{1}}
    ρ::Float64
    μ::Float64
    σ::Float64
    t::Float64
    u::Float64
    T::Float64
    v::Float64
end

B(t, P::DoubleWellPotentialAux) = @SMatrix [0.0]
β(t, P::DoubleWellPotentialAux) = ℝ{1}(0.0)
σ(t, P::DoubleWellPotentialAux) = ℝ{1}(P.σ)
dependsOnParams(::DoubleWellPotentialAux) = (3,)
constdiff(::DoubleWellPotentialAux) = true
b(t, x, P::DoubleWellPotentialAux) = B(t, P) * x + β(t, P)
a(t, P::DoubleWellPotentialAux) = σ(t, P) * σ(t, P)'
clone(P::DoubleWellPotentialAux, θ) = DoubleWellPotentialAux(θ..., P.t, P.u, P.T, P.v)
clone(P::DoubleWellPotentialAux, θ, 𝓣) = clone(P, (θ[1:2]..., 𝓣))
