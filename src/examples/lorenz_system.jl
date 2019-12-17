using Bridge
using StaticArrays, LinearAlgebra
import Bridge: b, σ, B, β, a, constdiff
const ℝ = SVector{N,T} where {N,T}

struct Lorenz{T} <: ContinuousTimeProcess{ℝ{3,T}}
    θ₁::T
    θ₂::T
    θ₃::T
    σ₁::T
    σ₂::T
    σ₃::T
    function Lorenz(θ₁::T, θ₂::T, θ₃::T, σ₁::T, σ₂::T, σ₃::T) where T
        new{T}(θ₁, θ₂, θ₃, σ₁, σ₂, σ₃)
    end
end

function b(t, x, P::Lorenz)
    ℝ{3}(P.θ₁*(x[2]-x[1]),
         P.θ₂*x[1] - x[2] - x[1]*x[3],
         x[1]*x[2] - P.θ₃*x[3])
end

function σ(t, x, P::Lorenz)
    @SMatrix[ P.σ₁ 0.0 0.0;
              0.0 P.σ₂ 0.0;
              0.0 0.0 P.σ₃]
end

constdiff(::Lorenz) = true
clone(P::Lorenz, θ) = Lorenz(θ...)
params(P::Lorenz) = [P.θ₁, P.θ₂, P.θ₃, P.σ₁, P.σ₂, P.σ₃]


struct LorenzAux{O,R,S1,S2} <: ContinuousTimeProcess{ℝ{3,R}}
    θ₁::R
    θ₂::R
    θ₃::R
    σ₁::R
    σ₂::R
    σ₃::R
    t::Float64
    u::S1
    T::Float64
    v::S2
    aux::Float64

    function LorenzAux(θ₁::R, θ₂::R, θ₃::R, σ₁::R, σ₂::R, σ₃::R, t, u::S1, T,
                       v::S2, ::O, aux=0.0) where {O,R,S1,S2}
        new{O,R,S1,S2}(θ₁, θ₂, θ₃, σ₁, σ₂, σ₃, t, u, T, v, aux)
    end
end

observables(::LorenzAux{O}) where O = O()


# Auxiliary diffusion when coordinates [1,2,3] are observed
# ---------------------------------------------------------
function B(t, P::LorenzAux{Val{(true,true,true)}})
    @SMatrix [ -P.θ₁  P.θ₁  0.0;
               P.θ₂-P.v[3]  -1.0  -P.v[1];
               P.v[2]  P.v[1]  -P.θ₃]
end

function β(t, P::LorenzAux{Val{(true,true,true)}})
    @SVector [0.0,
              P.v[1]*P.v[3],
              -P.v[1]*P.v[2]]
end

function σ(t, P::LorenzAux{Val{(true,true,true)}})
    @SMatrix[ P.σ₁ 0.0 0.0;
              0.0 P.σ₂ 0.0;
              0.0 0.0 P.σ₃]
end

a(t, P::LorenzAux) = σ(t,P) * σ(t, P)'
σ(t, x, P::LorenzAux) = σ(t, P)
b(t, x, P::LorenzAux) = B(t, P)*x + β(t, P)

constdiff(::LorenzAux) = true
clone(P::LorenzAux, θ) = LorenzAux(θ..., P.t, P.u, P.T, P.v, observables(P), P.aux)
clone(P::LorenzAux, θ, v) = LorenzAux(θ..., P.t, zero(v), P.T, v, observables(P), P.aux)
params(P::LorenzAux) = [P.θ₁, P.θ₂, P.θ₃, P.σ₁, P.σ₂, P.σ₃]
depends_on_params(::LorenzAux) = (1,2,3,4,5,6)


# Auxiliary diffusion when coordinates [1,2] are observed
# ---------------------------------------------------------
#observables(::LorenzAux{O}) where O = O()

function B(t, P::LorenzAux{Val{(true,true,false)}})
    @SMatrix [ -P.θ₁  P.θ₁  0.0;
               P.θ₂-P.aux  -1.0  -P.v[1];
               P.v[2]  P.v[1]  -P.θ₃]
end

function β(t, P::LorenzAux{Val{(true,true,false)}})
    @SVector [0.0,
              P.v[1]*P.aux,
              -P.v[1]*P.v[2]]
end

function σ(t, P::LorenzAux{Val{(true,true,false)}})
    @SMatrix[ P.σ₁ 0.0 0.0;
              0.0 P.σ₂ 0.0;
              0.0 0.0 P.σ₃]
end
