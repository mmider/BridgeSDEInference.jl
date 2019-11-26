using Bridge
using StaticArrays, LinearAlgebra
import Bridge: b, σ, B, β, a, constdiff
const ℝ = SVector{N,T} where {N,T}
import Base.resize!
using Interpolations

struct LorenzCV{T} <: ContinuousTimeProcess{ℝ{3,T}}
    θ₁::T
    θ₂::T
    θ₃::T
    σ::T
    function LorenzCV(θ₁::T, θ₂::T, θ₃::T, σ::T) where T
        new{T}(θ₁, θ₂, θ₃, σ)
    end
end

function b(t, x, P::LorenzCV)
    ℝ{3}(P.θ₁*(x[2]-x[1]),
         P.θ₂*x[1] - x[2] - x[1]*x[3],
         x[1]*x[2] - P.θ₃*x[3])
end

function σ(t, x, P::LorenzCV)
    @SMatrix[ P.σ 0.0 0.0;
              0.0 P.σ 0.0;
              0.0 0.0 P.σ]
end

constdiff(::LorenzCV) = true
clone(P::LorenzCV, θ) = LorenzCV(θ...)
params(P::LorenzCV) = [P.θ₁, P.θ₂, P.θ₃, P.σ]

nonhypo(P::LorenzCV, x) = x
@inline hypo_a_inv(P::LorenzCV, t, x) = SMatrix{3,3}(I)/P.σ^2
num_non_hypo(P::Type{<:LorenzCV}) = 3

phi(::Val{0}, t, x, P::LorenzCV) = (zero(x[1]), -x[2]-x[1]*x[3], x[1]*x[2])
phi(::Val{1}, t, x, P::LorenzCV) = (x[2]-x[1], zero(x[1]), zero(x[1]))
phi(::Val{2}, t, x, P::LorenzCV) = (zero(x[1]), x[1], zero(x[1]))
phi(::Val{3}, t, x, P::LorenzCV) = (zero(x[1]), zero(x[1]), -x[3])
phi(::Val{4}, t, x, P::LorenzCV) = (zero(x[1]), zero(x[1]), zero(x[1]))

struct LorenzCVAux{O,R,S1,S2,TI} <: ContinuousTimeProcess{ℝ{3,R}}
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

    function LorenzCVAux(θ₁::R, θ₂::R, θ₃::R, σ::R, t, u::S1, T,
                         v::S2, ::O, aux=0.0) where {O,R,S1,S2}
        λ = [1.0]
        X̄ = LinearInterpolation([t, T], zeros(ℝ{3,Float64}, 2), extrapolation_bc = Line())
        TI = typeof(X̄)
        new{O,R,S1,S2,TI}(θ₁, θ₂, θ₃, σ, t, u, T, v, aux, λ, X̄)
    end

    function LorenzCVAux(θ₁::R, θ₂::R, θ₃::R, σ::R, t, u::S1, T,
                         v::S2, ::Type{O}, aux, λ::Vector{Float64}, X̄::TI) where {O,R,S1,S2,TI}
        new{O,R,S1,S2,TI}(θ₁, θ₂, θ₃, σ, t, u, T, v, aux, λ, X̄)
    end

    function LorenzCVAux(θ₁::R, θ₂::R, θ₃::R, σ::R, t, u::S1, T,
                         v::S2, ::Type{Nothing}, aux, λ::Vector{Float64}, X̄::TI) where {R,S1,S2,TI}
        new{Val{(true,true,true)},R,S1,S2,TI}(θ₁, θ₂, θ₃, σ, t, u, T, v, aux, λ,
                                              X̄)
    end

    function LorenzCVAux(P::LorenzCVAux{O,R,S1,S2}, X̄::TI) where {O,R,S1,S2,TI}
        new{O,R,S1,S2,TI}(P.θ₁, P.θ₂, P.θ₃, P.σ, P.t, P.u, P.T, P.v, P.aux, P.λ,
                          X̄)
    end

    function LorenzCVAux(P::LorenzCVAux{Val{(false,true,true)},R,S1,S2}, X̄::TI
                         ) where {R,S1,S2,TI}
        new{Val{(false,true,true)},R,S1,S2,TI}(P.θ₁, P.θ₂, P.θ₃, P.σ, P.t, P.u,
                                             P.T, P.v, X̄(P.T)[1], P.λ, X̄)
    end
end

function recentre(P::LorenzCVAux, tt, X̄)
    itp = LinearInterpolation(collect(tt), X̄, extrapolation_bc = Line())
    LorenzCVAux(P, itp)
end

get_aux_flag(::LorenzCVAux{O}) where O = O

function update_λ!(P::LorenzCVAux, λ)
    @assert 0.0 ≤ λ ≤ 1.0
    P.λ[1] = λ
end


B(t, P::LorenzCVAux) = B₀(t, P)
β(t, P::LorenzCVAux) = β₀(t, P)
# NOTE uncomment if adaptation is to be used, currently commented out, because it is slow
#B(t, P::LorenzCVAux) = P.λ[1] * B₀(t, P) + (1.0-P.λ[1]) * B_bar(t, P)
#β(t, P::LorenzCVAux) = P.λ[1] * β₀(t, P) + (1.0-P.λ[1]) * β_bar(t, P)


function β_bar(t, P::LorenzCVAux)
    x = P.X̄(t)
    b_trgt(P, x) - J_b(P, x)*x
end

function B_bar(t, P::LorenzCVAux)
    x = P.X̄(t)
    J_b(P, x)
end

function b_trgt(P::LorenzCVAux, x)
    ℝ{3}(P.θ₁*(x[2]-x[1]),
         P.θ₂*x[1] - x[2] - x[1]*x[3],
         x[1]*x[2] - P.θ₃*x[3])
end

function J_b(P::LorenzCVAux, x)
    @SMatrix[ -P.θ₁  P.θ₁  0.0;
              (P.θ₂-x[3])  -1.0  -x[1];
              x[2]  x[1]  -P.θ₃]
end

function σ(t, P::LorenzCVAux)
    @SMatrix[ P.σ 0.0 0.0;
              0.0 P.σ 0.0;
              0.0 0.0 P.σ]
end


a(t, P::LorenzCVAux) = σ(t,P) * σ(t, P)'
σ(t, x, P::LorenzCVAux) = σ(t, P)
b(t, x, P::LorenzCVAux) = B(t, P)*x + β(t, P)

constdiff(::LorenzCVAux) = true

function clone(P::LorenzCVAux, θ)
    LorenzCVAux(θ..., P.t, P.u, P.T, P.v, get_aux_flag(P), P.aux, P.λ, P.X̄)
end

function clone(P::LorenzCVAux, θ, v, aux_flag)
    LorenzCVAux(θ..., P.t, zero(v), P.T, v, aux_flag, P.aux, P.λ, P.X̄)
end

params(P::LorenzCVAux) = [P.θ₁, P.θ₂, P.θ₃, P.σ]
depends_on_params(::LorenzCVAux) = (1,2,3,4)


# Auxiliary diffusion when coordinates [1,2,3] are observed
# ---------------------------------------------------------
function B₀(t, P::LorenzCVAux{Val{(true,true,true)}})
    @SMatrix [ -P.θ₁  P.θ₁  0.0;
               P.θ₂-P.v[3]  -1.0  -P.v[1];
               P.v[2]  P.v[1]  -P.θ₃]
end

function β₀(t, P::LorenzCVAux{Val{(true,true,true)}})
    @SVector [0.0,
              P.v[1]*P.v[3],
              -P.v[1]*P.v[2]]
end


# Auxiliary diffusion when coordinates [1,2] are observed
# ---------------------------------------------------------
function B₀(t, P::LorenzCVAux{Val{(true,true,false)}})
    @SMatrix [ -P.θ₁  P.θ₁  0.0;
               P.θ₂-P.aux  -1.0  -P.v[1];
               P.v[2]  P.v[1]  -P.θ₃]
end

function β₀(t, P::LorenzCVAux{Val{(true,true,false)}})
    @SVector [0.0,
              P.v[1]*P.aux,
              -P.v[1]*P.v[2]]
end


# Auxiliary diffusion when coordinates [2,3] are observed
# ---------------------------------------------------------
function B₀(t, P::LorenzCVAux{Val{(false,true,true)}})
    @SMatrix [ -P.θ₁  P.θ₁  0.0;
               P.θ₂-P.v[2]  -1.0  -P.aux;
               P.v[1]  P.aux  -P.θ₃]
end

function β₀(t, P::LorenzCVAux{Val{(false,true,true)}})
    @SVector [0.0,
              P.aux*P.v[2],
              -P.aux*P.v[1]]
end




#=

function B₀(t, P::LorenzCVAux{Val{(true,true,true)}})
    @SMatrix [ 0.0  0.0  0.0;
               0.0  0.0  0.0;
               0.0  0.0  0.0]
end

function β₀(t, P::LorenzCVAux{Val{(true,true,true)}})
    @SVector [0.0,
              0.0,
              0.0]
end

function B₀(t, P::LorenzCVAux{Val{(true,true,false)}})
    @SMatrix [ 0.0  0.0  0.0;
               0.0  0.0  0.0;
               0.0  0.0  0.0]
end

function β₀(t, P::LorenzCVAux{Val{(true,true,false)}})
    @SVector [0.0,
              0.0,
              0.0]
end

function B₀(t, P::LorenzCVAux{Val{(false,true,true)}})
    @SMatrix [ 0.0  0.0  0.0;
               0.0  0.0  0.0;
               0.0  0.0  0.0]
end

function β₀(t, P::LorenzCVAux{Val{(false,true,true)}})
    @SVector [0.0,
              0.0,
              0.0]
end

depends_on_params(::LorenzCVAux) = (4,)
=#
