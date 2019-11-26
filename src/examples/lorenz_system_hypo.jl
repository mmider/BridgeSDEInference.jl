using Bridge
using StaticArrays, LinearAlgebra
import Bridge: b, σ, B, β, a, constdiff
const ℝ = SVector{N,T} where {N,T}
import Base.resize!
using Interpolations

struct LorenzHypo{T} <: ContinuousTimeProcess{ℝ{3,T}}
    θ₁::T
    θ₂::T
    θ₃::T
    σ::T
    function LorenzHypo(θ₁::T, θ₂::T, θ₃::T, σ::T) where T
        new{T}(θ₁, θ₂, θ₃, σ)
    end
end

function b(t, x, P::LorenzHypo)
    ℝ{3}(P.θ₁*(x[2]-x[1]),
         P.θ₂*x[1] - x[2] - x[1]*x[3],
         x[1]*x[2] - P.θ₃*x[3])
end

function σ(t, x, P::LorenzHypo)
    @SMatrix[ 0.0 0.0;
              P.σ 0.0;
              0.0 P.σ]
end

constdiff(::LorenzHypo) = true
clone(P::LorenzHypo, θ) = LorenzHypo(θ...)
params(P::LorenzHypo) = [P.θ₁, P.θ₂, P.θ₃, P.σ]

nonhypo(P::LorenzHypo, x) = x[2:3]
@inline hypo_a_inv(P::LorenzHypo, t, x) = SMatrix{2,2}(I)/P.σ^2
num_non_hypo(P::Type{<:LorenzHypo}) = 2

phi(::Val{0}, t, x, P::LorenzHypo) = (-x[2]-x[1]*x[3], x[1]*x[2])
phi(::Val{1}, t, x, P::LorenzHypo) = (zero(x[1]), zero(x[1]))
phi(::Val{2}, t, x, P::LorenzHypo) = (x[1], zero(x[1]))
phi(::Val{3}, t, x, P::LorenzHypo) = (zero(x[1]), -x[3])
phi(::Val{4}, t, x, P::LorenzHypo) = (zero(x[1]), zero(x[1]))

struct LorenzHypoAux{O,R,S1,S2,TI} <: ContinuousTimeProcess{ℝ{3,R}}
    θ₁::R
    θ₂::R
    θ₃::R
    σ::R
    t::Float64
    u::S1
    T::Float64
    v::S2
    aux::Float64
    λ::Vector{Float64}
    X̄::TI

    function LorenzHypoAux(θ₁::R, θ₂::R, θ₃::R, σ::R, t, u::S1, T,
                         v::S2, ::O, aux=0.0) where {O,R,S1,S2}
        λ = [1.0]
        X̄ = LinearInterpolation([t, T], zeros(ℝ{3,Float64}, 2), extrapolation_bc = Line())
        TI = typeof(X̄)
        new{O,R,S1,S2,TI}(θ₁, θ₂, θ₃, σ, t, u, T, v, aux, λ, X̄)
    end

    function LorenzHypoAux(θ₁::R, θ₂::R, θ₃::R, σ::R, t, u::S1, T,
                         v::S2, ::Type{O}, aux, λ::Vector{Float64}, X̄::TI) where {O,R,S1,S2,TI}
        new{O,R,S1,S2,TI}(θ₁, θ₂, θ₃, σ, t, u, T, v, aux, λ, X̄)
    end

    function LorenzHypoAux(θ₁::R, θ₂::R, θ₃::R, σ::R, t, u::S1, T,
                         v::S2, ::Type{Nothing}, aux, λ::Vector{Float64}, X̄::TI) where {R,S1,S2,TI}
        new{Val{(true,true,true)},R,S1,S2,TI}(θ₁, θ₂, θ₃, σ, t, u, T, v, aux, λ,
                                              X̄)
    end

    function LorenzHypoAux(P::LorenzHypoAux{O,R,S1,S2}, X̄::TI) where {O,R,S1,S2,TI}
        new{O,R,S1,S2,TI}(P.θ₁, P.θ₂, P.θ₃, P.σ, P.t, P.u, P.T, P.v, P.aux, P.λ,
                          X̄)
    end

    function LorenzHypoAux(P::LorenzHypoAux{Val{(false,true,true)},R,S1,S2}, X̄::TI
                         ) where {R,S1,S2,TI}
        new{Val{(false,true,true)},R,S1,S2,TI}(P.θ₁, P.θ₂, P.θ₃, P.σ, P.t, P.u,
                                             P.T, P.v, X̄(P.T)[1], P.λ, X̄)
    end
end

function recentre(P::LorenzHypoAux, tt, X̄)
    itp = LinearInterpolation(collect(tt), X̄, extrapolation_bc = Line())
    LorenzHypoAux(P, itp)
end

get_aux_flag(::LorenzHypoAux{O}) where O = O

function update_λ!(P::LorenzHypoAux, λ)
    @assert 0.0 ≤ λ ≤ 1.0
    P.λ[1] = λ
end


B(t, P::LorenzHypoAux) = B₀(t, P)
β(t, P::LorenzHypoAux) = β₀(t, P)
# NOTE uncomment if adaptation is to be used, currently commented out, because it is slow
#B(t, P::LorenzHypoAux) = P.λ[1] * B₀(t, P) + (1.0-P.λ[1]) * B_bar(t, P)
#β(t, P::LorenzHypoAux) = P.λ[1] * β₀(t, P) + (1.0-P.λ[1]) * β_bar(t, P)


function β_bar(t, P::LorenzHypoAux)
    x = P.X̄(t)
    b_trgt(P, x) - J_b(P, x)*x
end

function B_bar(t, P::LorenzHypoAux)
    x = P.X̄(t)
    J_b(P, x)
end

function b_trgt(P::LorenzHypoAux, x)
    ℝ{3}(P.θ₁*(x[2]-x[1]),
         P.θ₂*x[1] - x[2] - x[1]*x[3],
         x[1]*x[2] - P.θ₃*x[3])
end

function J_b(P::LorenzHypoAux, x)
    @SMatrix[ -P.θ₁  P.θ₁  0.0;
              (P.θ₂-x[3])  -1.0  -x[1];
              x[2]  x[1]  -P.θ₃]
end

function σ(t, P::LorenzHypoAux)
    @SMatrix[ 0.0 0.0;
              P.σ 0.0;
              0.0 P.σ]
end


a(t, P::LorenzHypoAux) = σ(t,P) * σ(t, P)'
σ(t, x, P::LorenzHypoAux) = σ(t, P)
b(t, x, P::LorenzHypoAux) = B(t, P)*x + β(t, P)

constdiff(::LorenzHypoAux) = true

function clone(P::LorenzHypoAux, θ)
    LorenzHypoAux(θ..., P.t, P.u, P.T, P.v, get_aux_flag(P), P.aux, P.λ, P.X̄)
end

function clone(P::LorenzHypoAux, θ, v, aux_flag)
    LorenzHypoAux(θ..., P.t, zero(v), P.T, v, aux_flag, P.aux, P.λ, P.X̄)
end

params(P::LorenzHypoAux) = [P.θ₁, P.θ₂, P.θ₃, P.σ]
depends_on_params(::LorenzHypoAux) = (1,2,3,4)


# Auxiliary diffusion when coordinates [1,2,3] are observed
# ---------------------------------------------------------
function B₀(t, P::LorenzHypoAux{Val{(true,true,true)}})
    @SMatrix [ -P.θ₁  P.θ₁  0.0;
               P.θ₂-P.v[3]  -1.0  -P.v[1];
               P.v[2]  P.v[1]  -P.θ₃]
end

function β₀(t, P::LorenzHypoAux{Val{(true,true,true)}})
    @SVector [0.0,
              P.v[1]*P.v[3],
              -P.v[1]*P.v[2]]
end


# Auxiliary diffusion when coordinates [1,2] are observed
# ---------------------------------------------------------
function B₀(t, P::LorenzHypoAux{Val{(true,true,false)}})
    @SMatrix [ -P.θ₁  P.θ₁  0.0;
               P.θ₂-P.aux  -1.0  -P.v[1];
               P.v[2]  P.v[1]  -P.θ₃]
end

function β₀(t, P::LorenzHypoAux{Val{(true,true,false)}})
    @SVector [0.0,
              P.v[1]*P.aux,
              -P.v[1]*P.v[2]]
end


# Auxiliary diffusion when coordinates [2,3] are observed
# ---------------------------------------------------------
function B₀(t, P::LorenzHypoAux{Val{(false,true,true)}})
    @SMatrix [ -P.θ₁  P.θ₁  0.0;
               P.θ₂-P.v[2]  -1.0  -P.aux;
               P.v[1]  P.aux  -P.θ₃]
end

function β₀(t, P::LorenzHypoAux{Val{(false,true,true)}})
    @SVector [0.0,
              P.aux*P.v[2],
              -P.aux*P.v[1]]
end




#=

function B₀(t, P::LorenzHypoAux{Val{(true,true,true)}})
    @SMatrix [ 0.0  0.0  0.0;
               0.0  0.0  0.0;
               0.0  0.0  0.0]
end

function β₀(t, P::LorenzHypoAux{Val{(true,true,true)}})
    @SVector [0.0,
              0.0,
              0.0]
end

function B₀(t, P::LorenzHypoAux{Val{(true,true,false)}})
    @SMatrix [ 0.0  0.0  0.0;
               0.0  0.0  0.0;
               0.0  0.0  0.0]
end

function β₀(t, P::LorenzHypoAux{Val{(true,true,false)}})
    @SVector [0.0,
              0.0,
              0.0]
end

function B₀(t, P::LorenzHypoAux{Val{(false,true,true)}})
    @SMatrix [ 0.0  0.0  0.0;
               0.0  0.0  0.0;
               0.0  0.0  0.0]
end

function β₀(t, P::LorenzHypoAux{Val{(false,true,true)}})
    @SVector [0.0,
              0.0,
              0.0]
end

depends_on_params(::LorenzHypoAux) = (4,)
=#
