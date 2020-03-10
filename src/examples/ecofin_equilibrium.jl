using Bridge
using StaticArrays
using LinearAlgebra
import Bridge: b, σ, B, β, a, constdiff
const ℝ = SVector{N, T} where {N, T}

"""
    EcoFinEq <: ContinuousTimeProcess{ℝ{6, T}}
structure defining an economical and financial equilibium system described in
B. J. Christensen et al., Estimating dynamic equilibrium models using mixed frequencymacro and financial data
"""
struct EcoFinEq{T} <: ContinuousTimeProcess{ℝ{3, T}}
    ρ::T
    δ::T
    γ::T
    κ::T
    η::T
    σ::T
    function EcoFinEq(ρ::T,δ::T, γ::T, κ::T, η::T, σ::T) where T
        new{T}(ρ, δ, γ, κ, η, σ)
    end
end

function b(t, x, P::EcoFinEq{T}) where T
    ℝ{3, T}(x[3] - P.ρ - P.δ - 0.5*P.σ^2,
    P.κ*P.γ/x[3] - 0.5*P.η^2/x[3]^2 + x[3] - P.κ - P.ρ - P.δ - 0.5*P.σ^2,
    P.κ*(P.γ - x[3])
    )
end


function σ(t, x, P::EcoFinEq{T}) where T
    @SMatrix    [P.σ 0.0;
                P.σ  P.η/x[3];
                0.0 P.η]
end


constdiff(::EcoFinEq) = false
clone(::EcoFinEq, θ) = JRNeuralDiffusion3n(θ...)
#Static vector
params(P::EcoFinEq) = [P.ρ, P.δ, P.γ, P.κ, P.η, P.σ]
param_names(::EcoFinEq) = (:ρ, :δ, :γ, :κ, :η, :σ)

##Conjugate Step
#TODO
# @inline hypo_a_inv(P::JRNeuralDiffusion3n, t, x) = SMatrix{3,3}(Diagonal([P.σx^(-2), P.σy^(-2), P.σz^(-2)]))
# nonhypo(P::JRNeuralDiffusion3n, x) = x[4:6]
# num_non_hypo(P::Type{<:JRNeuralDiffusion3n}) = 3

##auxiliary process
struct EcoFinEqAux{R, S1, S2} <: ContinuousTimeProcess{ℝ{3, R}}
    ρ::R
    δ::R
    γ::R
    κ::R
    η::R
    σ::R
    t::Float64
    u::S1
    T::Float64
    v::S2
    function EcoFinEqAux(ρ::R, δ::R, γ::R, κ::R, η::R, σ::R, t::Float64, u::S1, T::Float64, v::S2) where {R, S1, S2}
        new{R, S1, S2}(ρ,δ, γ, κ, η, σ, t, u, T, v)
    end
end

function B(t, P::EcoFinEqAux)
    @SMatrix    [0.0 0.0 1.0;
                 0.0 0.0 1.0;
                 0.0 0.0 -P.κ]
end



function β(t,  P::EcoFinEqAux)
    ℝ{3}(-P.ρ - P.δ - 0.5*P.σ^2,
    P.κ*P.γ/P.v[end] - 0.5*P.η^2/P.v[end]^2 - P.κ - P.ρ - P.δ - 0.5*P.σ^2,
    P.κ*P.γ
    )
end
# function β(t,  P::EcoFinEqAux)
#     ℝ{3}(-P.ρ - P.δ - 0.5*P.σ^2,
#     P.κ - 0.5*P.η^2/P.γ^2 - P.κ - P.ρ - P.δ - 0.5*P.σ^2,
#     P.κ*P.γ
#     )
# end





# function σ(t, P::EcoFinEqAux)
#     @SMatrix    [P.σ 0.0;
#                 P.σ  P.η/P.γ;
#                 0.0 P.η]
# end

function σ(t, P::EcoFinEqAux)
    @SMatrix    [P.σ 0.0;
                P.σ  P.η/P.v[end];
                0.0 P.η]
end

b(t, x, P::EcoFinEqAux) = B(t,P) * x + β(t,P)
a(t, P::EcoFinEqAux) = σ(t,P) * σ(t, P)'

constdiff(::EcoFinEqAux) = true
clone(P::EcoFinEqAux, θ) = EcoFinEqAux(θ..., P.t, P.u, P.T, P.v)
clone(P::EcoFinEqAux, θ, v) = EcoFinEqAux(θ..., P.t, zero(v), P.T, v)
params(P::EcoFinEqAux) = [P.ρ, P.δ, P.γ, P.κ, P.η, P.σ]
param_names(P::EcoFinEqAux) = (:ρ, :δ, :γ, :κ, :η, :σ)
depends_on_params(::EcoFinEqAux) = (1, 2, 3, 4, 5, 6)
