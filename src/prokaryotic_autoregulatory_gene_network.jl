# Example taken from Golithly and Wilkinson 2005 `Bayesian Inference for Stochastic
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

function b(t, x, P::Prokaryote{T}) where T # x <-> (RNA, P, P₂, DNA)
    ℝ{4}(P.c₃*x[4] - P.c₇*x[1],
         P.c₄*x[1] + 2.0*P.c₆*x[3] - P.c₅*x[2]*(x[2]-1)-P.c₈*x[2],
         P.c₂*(P.K - x[4]) + 0.5*P.c₅*x[2]*(x[2]-1) - P.c₁*x[3]*x[4] - P.c₆*x[3],
         P.c₂*(P.K - x[4]) - P.c₁*x[3]*x[4])
end

function σ(t, x, P::Prokaryote{T}) where T
    σσᵀ = a(t, x, P)
    cholesky(σσᵀ).U'
end

#=
 c₃DNA+c₇RNA |         0          |           0          |          0
--------------------------------------------------------------------------------
     0       |  c₄RNA+2c₅P(P-1)   |  -c₅P(P-1)-2c₆P₂     |          0
             |      +4c₆P₂+c₈P    |                      |
--------------------------------------------------------------------------------
     0       | -c₅P(P-1)-2c₆P₂    |  c₁P₂DNA+c₂(K-DNA)   |   c₁P₂DNA
             |                    |    +0.5c₅P(P-1)+c₆P₂ |    +c₂(K-DNA)
--------------------------------------------------------------------------------
     0       |         0          |   c₁P₂DNA+c₂(K-DNA)  |    c₁P₂DNA
                                                                +c₂(K-DNA)
=#
# x <-> (RNA, P, P₂, DNA)
function a(t, x, P::Prokaryote{T}) where T
    _a₃₂ = P.c₅*x[2]*(x[2]-1.0)+2.0*P.c₆*x[3]
    a₄₄ = P.c₁*x[3]*x[4]+P.c₂*(P.K-x[4])

    @SMatrix [P.c₃*x[4]+P.c₇*x[1]  0.0  0.0  0.0;
              0.0  P.c₄*x[1]+2.0*_a₃₂+P.c₈*x[2]  -_a₃₂  0.0;
              0.0  -_a₃₂  a₄₄+0.5*_a₃₂  a₄₄;
              0.0  0.0  a₄₄  a₄₄]
end

domain(P::Prokaryote) = BoundedDomain( LowerBoundedDomain((0.0, 1.0, 0.0, 0.0), (1,2,3,4)),
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

function B(t, P::ProkaryoteAux{Val{(true,true,true,true)}})
    @SMatrix [-P.c₇  0.0  0.0  P.c₃;
              P.c₄  P.c₅-P.c₈-2*c₅*P.v[2]  2.0*P.c₆  0.0;
              0.0  P.c₅*P.v[2]-0.5*P.c₅  -P.c₁*P.v[4]-P.c₆  -P.c₂-P.c₁*P.v[3];
              0.0  0.0  -P.c₁*P.v[4]  -P.c₂-P.c₁*P.v[3]]
end

function β(t, P::ProkaryoteAux{Val{(true,true,true,true)}})
    @SVector [0.0,
              P.c₅*P.v[2]^2,
              P.c₂*P.K + P.c₁*P.v[3]*P.v[4] - 0.5*P.c₅*P.v[2]^2,
              P.c₂*P.K + P.c₁*P.v[3]*P.v[4]]
end
