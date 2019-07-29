using Bridge
using StaticArrays
import Bridge: b, σ, B, β, a, constdiff
const ℝ = SVector{N,T} where {N,T}

"""
    RadialOU{T} <: ContinuousTimeProcess{ℝ{1,T}}

Struct defining Radial Ornstein Uhlenbeck processs
"""
struct RadialOU{T} <: ContinuousTimeProcess{ℝ{1,T}}
    η::T
    σ::T
    RadialOU(η::T, σ::T) where T = new{T}(η, σ)
end


b(t, x, P::RadialOU) = ℝ{1}(-P.η*x[1] + 0.5*P.σ^2/x[1])
σ(t, x, P::RadialOU) = ℝ{1}(P.σ^2)

domain(::RadialOU{T}) where T = LowerBoundedDomain((zero(T),), (1,))
constdiff(::RadialOU) = true
clone(P::RadialOU, θ) = RadialOU(θ...)
params(P::RadialOU) = [P.η, P.σ]


"""
    RadialOUAux{S} <: ContinuousTimeProcess{ℝ{1,S}}
"""
struct RadialOUAux{S} <: ContinuousTimeProcess{ℝ{1,S}}
    trgtDomain::LowerBoundedDomain{S,1}
    η::S
    σ::S
    t::Float64
    u::S
    T::Float64
    v::S

    function RadialOUAux(η::S, σ::S, t::Float64, u::S, T::Float64, v::S) where S
        domain = LowerBoundedDomain((zero(S),), (1,))
        new{S}(domain, η, σ, t, u, T, v)
    end

    function RadialOUAux(domain::LowerBoundedDomain{S,1}, η::S, σ::S,
                         t::Float64, u::S, T::Float64, v::S) where S
        new{S}(domain, η, σ, t, u, T, v)
    end
end

B(t, P::RadialOUAux) = @SMatrix[0.0]
β(t, P::RadialOUAux) = ℝ{1}(0.0)
σ(t, P::RadialOUAux) = ℝ{1}(P.σ^2)
dependsOnParams(::RadialOUAux) = (2,)
constdiff(::RadialOUAux) = true
b(t, x, P::RadialOUAux) = B(t,P)*x + β(t,P)
a(t, P::RadialOUAux) = σ(t,P)*σ(t,P)'
clone(P::RadialOUAux, θ) = RadialOUAux(P.trgtDomain, θ..., P.t, P.u, P.T, P.v)
params(P::RadialOUAux) = [P.η, P.σ]
