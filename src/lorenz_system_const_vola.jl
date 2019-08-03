using Bridge
using StaticArrays, LinearAlgebra
import Bridge: b, σ, B, β, a, constdiff
const ℝ = SVector{N,T} where {N,T}

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


struct LorenzCVAux{O,R,S1,S2} <: ContinuousTimeProcess{ℝ{3,R}}
    θ₁::R
    θ₂::R
    θ₃::R
    σ::R
    t::Float64
    u::S1
    T::Float64
    v::S2
    aux::Float64

    function LorenzCVAux(θ₁::R, θ₂::R, θ₃::R, σ::R, t, u::S1, T,
                       v::S2, ::O, aux=0.0) where {O,R,S1,S2}
        new{O,R,S1,S2}(θ₁, θ₂, θ₃, σ, t, u, T, v, aux)
    end
end

observables(::LorenzCVAux{O}) where O = O()


# Auxiliary diffusion when coordinates [1,2,3] are observed
# ---------------------------------------------------------
function B(t, P::LorenzCVAux{Val{(true,true,true)}})
    @SMatrix [ -P.θ₁  P.θ₁  0.0;
               P.θ₂-P.v[3]  -1.0  -P.v[1];
               P.v[2]  P.v[1]  -P.θ₃]
end

function β(t, P::LorenzCVAux{Val{(true,true,true)}})
    @SVector [0.0,
              P.v[1]*P.v[3],
              -P.v[1]*P.v[2]]
end

function σ(t, P::LorenzCVAux{Val{(true,true,true)}})
    @SMatrix[ P.σ 0.0 0.0;
              0.0 P.σ 0.0;
              0.0 0.0 P.σ]
end

a(t, P::LorenzCVAux) = σ(t,P) * σ(t, P)'
σ(t, x, P::LorenzCVAux) = σ(t, P)
b(t, x, P::LorenzCVAux) = B(t, P)*x + β(t, P)

constdiff(::LorenzCVAux) = true
clone(P::LorenzCVAux, θ) = LorenzCVAux(θ..., P.t, P.u, P.T, P.v, observables(P), P.aux)
clone(P::LorenzCVAux, θ, v) = LorenzCVAux(θ..., P.t, zero(v), P.T, v, observables(P), P.aux)
params(P::LorenzCVAux) = [P.θ₁, P.θ₂, P.θ₃, P.σ]
dependsOnParams(::LorenzCVAux) = (1,2,3,4,5,6)


# Auxiliary diffusion when coordinates [1,2] are observed
# ---------------------------------------------------------
observables(::LorenzCVAux{O}) where O = O()

function B(t, P::LorenzCVAux{Val{(true,true,false)}})
    @SMatrix [ -P.θ₁  P.θ₁  0.0;
               P.θ₂-P.aux  -1.0  -P.v[1];
               P.v[2]  P.v[1]  -P.θ₃]
end

function β(t, P::LorenzCVAux{Val{(true,true,false)}})
    @SVector [0.0,
              P.v[1]*P.aux,
              -P.v[1]*P.v[2]]
end

function σ(t, P::LorenzCVAux{Val{(true,true,false)}})
    @SMatrix[ P.σ 0.0 0.0;
              0.0 P.σ 0.0;
              0.0 0.0 P.σ]
end