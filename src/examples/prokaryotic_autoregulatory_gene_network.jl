# Example taken from Golightly and Wilkinson 2005 `Bayesian Inference for Stochastic
# Kinetic Models Using a Diffusion Approximation`

using Bridge
using StaticArrays, LinearAlgebra
import Bridge: b, σ, B, β, a, constdiff

struct Prokaryote{T} <: ContinuousTimeProcess{ℝ{4,T}}
    c₁::T
    c₂::T
    c₃::T
    c₄::T
    c₅::T
    c₆::T
    c₇::T
    c₈::T
    K::Float64

    function Prokaryote(c₁::T, c₂::T, c₃::T, c₄::T, c₅::T, c₆::T, c₇::T,
                        c₈::T, K) where T
        new{T}(c₁, c₂, c₃, c₄, c₅, c₆, c₇, c₈, K)
    end
end

# x <-> (RNA, P, P₂, DNA)
@inline _aux₁(x, P::Prokaryote) = P.c₅*x[2]*(x[2]-1)
@inline _aux₂(x, P::Prokaryote) = P.c₆*x[3]
@inline _aux₃(x, P::Prokaryote) = P.c₂*(P.K - x[4])
@inline _aux₄(x, P::Prokaryote) = P.c₁*x[3]*x[4]
@inline _aux₁₂(x, P::Prokaryote) = 2.0*_aux₂(x, P) - _aux₁(x, P)
@inline _aux₃₄(x, P::Prokaryote) = _aux₃(x, P) - _aux₄(x, P)

function b(t, x, P::Prokaryote{T}) where T
    k₁ = _aux₁₂(x, P)
    k₂ = _aux₃₄(x, P)
    ℝ{4}(P.c₃*x[4] - P.c₇*x[1], P.c₄*x[1] + k₁ - P.c₈*x[2], k₂ - 0.5 * k₁, k₂)
end

function _σ_prokaryote(x, P)
    k₁ = sqrt(0.5*_aux₁(x, P))
    k₂ = sqrt(_aux₂(x, P))
    k₃ = sqrt(_aux₃(x, P))
    k₄ = sqrt(_aux₄(x, P))
    _O = zero(x[1])

    @SMatrix [_O  _O  sqrt(P.c₃*x[4]) _O _O _O -sqrt(P.c₇*x[1]) _O;
              _O _O _O sqrt(P.c₄*x[1]) -2.0*k₁ 2.0*k₂ _O -sqrt(P.c₈*x[3]);
              -k₄ k₃ _O _O k₁ -k₂ _O _O;
              -k₄ k₃ _O _O _O _O _O _O]
end

function _a_prokaryote(x, P)
    k₁ = _aux₁(x, P) + 2.0*_aux₂(x, P)
    k₂ = _aux₃(x, P) + _aux₄(x, P)
    s₁ = P.c₃*x[4]+P.c₇*x[1]
    s₂ = P.c₄*x[1]+P.c₈*x[3]
    _O = zero(x[1])

    @SMatrix [ s₁ _O _O _O;
              _O s₂+2*k₁ -k₁ _O;
              _O -k₁ k₁+k₂ k₂;
              _O _O k₂ k₂]
end

σ(t, x, P::Prokaryote) = _σ_prokaryote(x, P)
a(t, x, P::Prokaryote) = _a_prokaryote(x, P)
domain(P::Prokaryote) = BoundedDomain( LowerBoundedDomain((0.0, 1.0, 0.0, 0.0),
                                                          (1,2,3,4)),
                                       UpperBoundedDomain((P.K,), (4,)) )
constdiff(::Prokaryote) = false
clone(P::Prokaryote, θ) = Prokaryote(θ..., P.K)
params(P::Prokaryote) = [P.c₁, P.c₂, P.c₃, P.c₄, P.c₅, P.c₆, P.c₇, P.c₈]

struct ProkaryoteAux{O,R,S1,S2} <: ContinuousTimeProcess{ℝ{4,R}}
    c₁::R
    c₂::R
    c₃::R
    c₄::R
    c₅::R
    c₆::R
    c₇::R
    c₈::R
    K::Float64
    t::Float64
    u::S1
    T::Float64
    v::S2

    function ProkaryoteAux(c₁::R, c₂::R, c₃::R, c₄::R, c₅::R, c₆::R, c₇::R,
                           c₈::R, K, t, u::S1, T, v::S2, ::O) where {O,R,S1,S2}
        new{O,R,S1,S2}(c₁, c₂, c₃, c₄, c₅, c₆, c₇, c₈, K, t, u, T, v)
    end
end

observables(::ProkaryoteAux{O}) where O = O()

function B(t, P::ProkaryoteAux{Val{(true,true,true,true)}})
    @SMatrix [-P.c₇  0.0  0.0  P.c₃;
              P.c₄  P.c₅-P.c₈-2*P.c₅*P.v[2]  2.0*P.c₆  0.0;
              0.0  P.c₅*P.v[2]-0.5*P.c₅  -P.c₁*P.v[4]-P.c₆  -P.c₂-P.c₁*P.v[3];
              0.0  0.0  -P.c₁*P.v[4]  -P.c₂-P.c₁*P.v[3]]
end

function β(t, P::ProkaryoteAux{Val{(true,true,true,true)}})
    @SVector [0.0,
              P.c₅*P.v[2]^2,
              P.c₂*P.K + P.c₁*P.v[3]*P.v[4] - 0.5*P.c₅*P.v[2]^2,
              P.c₂*P.K + P.c₁*P.v[3]*P.v[4]]
end

σ(t, P::ProkaryoteAux{Val{(true,true,true,true)}}) = _σ_prokaryote(P.v, P)
a(t, P::ProkaryoteAux{Val{(true,true,true,true)}}) = _a_prokaryote(P.V, P)
a(t, x, P::ProkaryoteAux) = a(t, P)
σ(t, x, P::ProkaryoteAux) = σ(t, P)
b(t, x, P::ProkaryoteAux) = B(t, P)*x + β(t, P)

constdiff(::ProkaryoteAux) = true
clone(P::ProkaryoteAux, θ) = ProkaryoteAux(θ..., P.K, P.t, P.u, P.T, P.v, observables(P))
clone(P::ProkaryoteAux, θ, v) = ProkaryoteAux(θ..., P.K, P.t, zero(v), P.T, v, observables(P))
params(P::ProkaryoteAux) = [P.c₁, P.c₂, P.c₃, P.c₄, P.c₅, P.c₆, P.c₇, P.c₈]
dependsOnParams(::ProkaryoteAux) = (1,2,3,4,5,6,7,8)
