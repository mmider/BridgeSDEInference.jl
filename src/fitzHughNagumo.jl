using Bridge
using StaticArrays
import Bridge: b, σ, B, β, a, constdiff
const ℝ = SVector{N,T} where {N,T}

print("Chosen parametrisation: ", parametrisation, "\n")

"""
    FitzhughDiffusion <: ContinuousTimeProcess{ℝ{2}}

Struct defining FitzHugh-Nagumo model
"""
struct FitzhughDiffusion{T} <: ContinuousTimeProcess{ℝ{2,T}}
    ϵ::T
    s::T
    γ::T
    β::T
    σ::T
end

if parametrisation == :regular
    b(t, x, P::FitzhughDiffusion) = ℝ{2}((x[1]-x[2]-x[1]^3+P.s)/P.ϵ,
                                         P.γ*x[1]-x[2] +P.β)
    σ(t, x, P::FitzhughDiffusion) = ℝ{2}(0.0, P.σ)
elseif parametrisation in (:simpleAlter, :complexAlter)
    function b(t, x, P::FitzhughDiffusion)
        ℝ{2}(x[2], -( (P.γ-1.0)*x[1] + x[1]^3 + P.ϵ*x[2] - P.s + P.β
                      + (3.0*x[1]^2 - 1.0)*x[2])/P.ϵ )
    end
    σ(t, x, P::FitzhughDiffusion) = ℝ{2}(0.0, P.σ/P.ϵ)
elseif parametrisation in (:simpleConjug, :complexConjug)
    function b(t, x, P::FitzhughDiffusion)
        ℝ{2}(x[2], ((P.ϵ - P.γ)*x[1] - P.ϵ*(x[1]^3 + (3.0*x[1]^2 - 1.0)*x[2])
                    + P.s - P.β - x[2]))
    end
    σ(t, x, P::FitzhughDiffusion) = ℝ{2}(0.0, P.σ)


    """
        φ(::Val{T}, args...)

    Compute the φ function appearing in the Girsanov formula and needed for
    sampling from the full conditional distribution of the parameters (whose
    indices are specified by the `Val`) conditional on the path,
    observations and other parameters.
    """
    @generated function φ(::Val{T}, args...) where T
        z = Expr(:tuple, (:(phi(Val($i), args...)) for i in 1:length(T) if T[i])...)
        return z
    end

    """
        φᶜ(::Val{T}, args...)

    Compute the φᶜ function appearing in the Girsanov formula. This function
    complements φ.
    """
    @generated function φᶜ(::Val{T}, args...) where T
        z = Expr(:tuple, (:(phi(Val($i), args...)) for i in 0:length(T) if i==0 || !T[i])...)
        return z
    end

    phi(::Val{0}, t, x, P::FitzhughDiffusion) = -x[2]
    phi(::Val{1}, t, x, P::FitzhughDiffusion) = x[1]-x[1]^3+(1-3*x[1]^2)*x[2]
    phi(::Val{2}, t, x, P::FitzhughDiffusion) = one(x[1])
    phi(::Val{3}, t, x, P::FitzhughDiffusion) = -x[1]
    phi(::Val{4}, t, x, P::FitzhughDiffusion) = zero(x[1])
    phi(::Val{5}, t, x, P::FitzhughDiffusion) = zero(x[1])
end


constdiff(::FitzhughDiffusion) = true

clone(::FitzhughDiffusion, θ) = FitzhughDiffusion(θ...)

params(P::FitzhughDiffusion) = [P.ϵ, P.s, P.γ, P.β, P.σ]

"""
    regularToAlter(x, ϵ, offset=0)
Transform point from observation under :regular parametrisation to the one under
:alter(...) parametrisation
"""
function regularToAlter(x, ϵ, offset=0)
    ℝ{2}(x[1], (x[1] - x[1]^3 - x[2] + offset) / ϵ)
end

"""
    alterToRegular(x, ϵ, offset=0)
Transform point from observation under :alter(...) parametrisation to the one
under :regular parametrisation
"""
function alterToRegular(x, ϵ, offset=0)
    ℝ{2}(x[1], x[1] - x[1]^3 - x[2]*ϵ + offset)
end

"""
    regularToConjug(x, ϵ, offset=0)
Transform point from observation under :regular parametrisation to the one under
:conjug(...) parametrisation
"""
function regularToConjug(x, ϵ, offset=0)
    ℝ{2}(x[1], (x[1] - x[1]^3 - x[2] + offset) * ϵ)
end

"""
    conjugToRegular(x, ϵ, offset=0)
Transform point from observation under :conjug(...) parametrisation to the one
under :regular parametrisation
"""
function conjugToRegular(x, ϵ, offset=0)
    ℝ{2}(x[1], x[1] - x[1]^3 - x[2]/ϵ + offset)
end


"""
    struct FitzhughDiffusionAux <: ContinuousTimeProcess{ℝ{2}}

Struct defining proposal diffusion (proposal for sampling from FitzHugh-Nagumo
diffusion)
"""
struct FitzhughDiffusionAux{T,S} <: ContinuousTimeProcess{ℝ{2,T}}
    ϵ::T
    s::T
    γ::T
    β::T
    σ::T
    t::Float64
    u::S
    T::Float64
    v::S
end

if parametrisation == :regular
    B(t, P::FitzhughDiffusionAux) = @SMatrix [1/P.ϵ-3*P.v^2/P.ϵ  -1/P.ϵ;
                                              P.γ -1.0] #2.5 <=> P.γ
    β(t, P::FitzhughDiffusionAux) = ℝ{2}(P.s/P.ϵ+2*P.v^3/P.ϵ, P.β) # P.s/P.ϵ<=>0.0
    σ(t, P::FitzhughDiffusionAux) = ℝ{2}(0.0, P.σ)

    """
        dependsOnParams(::FitzhughDiffusionAux)

    Declare which parameters (1=>`ϵ`, 2=>`s`, 3=>`γ`, 4=>`β`, 5=>`σ`) the
    auxiliary diffusion depends upon. Used for finding out which parameter
    update requires also updating the values of the grid of `H`'s and `r`'s.
    """
    dependsOnParams(::FitzhughDiffusionAux) = (1, 2, 3, 4, 5)
elseif parametrisation == :simpleAlter
    B(t, P::FitzhughDiffusionAux) = @SMatrix [0.0  1.0; 0.0 0.0]
    β(t, P::FitzhughDiffusionAux) = ℝ{2}(0.0, 0.0)
    σ(t, P::FitzhughDiffusionAux) = ℝ{2}(0.0, P.σ/P.ϵ)
    dependsOnParams(::FitzhughDiffusionAux) = (1, 5)
elseif parametrisation == :complexAlter
    B(t, P::FitzhughDiffusionAux) = @SMatrix [0.0  1.0;
                                (1.0-P.γ-3.0*P.v[1]^2)/P.ϵ (1.0-P.ϵ-3.0*P.v[1]^2)/P.ϵ]
    β(t, P::FitzhughDiffusionAux) = ℝ{2}(0.0, (2*P.v[1]^3+P.s-P.β)/P.ϵ)#P.s=>0.0
    σ(t, P::FitzhughDiffusionAux) = ℝ{2}(0.0, P.σ/P.ϵ)
    dependsOnParams(::FitzhughDiffusionAux) = (1, 2, 3, 4, 5)

    function B(t, P::FitzhughDiffusionAux{T,SArray{Tuple{2},Float64,1,2}}) where T
        @SMatrix [0.0  1.0;
                  (1.0-P.γ-3.0*P.v[1]^2-6*P.v[1]*P.v[2])/P.ϵ (1.0-P.ϵ-3.0*P.v[1]^2)/P.ϵ]
    end
    function β(t, P::FitzhughDiffusionAux{T,SArray{Tuple{2},Float64,1,2}}) where T
        ℝ{2}(0.0, (2*P.v[1]^3+P.s-P.β+6*P.v[1]^2*P.v[2])/P.ϵ)#check later
    end
elseif parametrisation == :simpleConjug
    B(t, P::FitzhughDiffusionAux) = @SMatrix [0.0  1.0; 0.0 0.0]
    β(t, P::FitzhughDiffusionAux) = ℝ{2}(0.0, 0.0)
    σ(t, P::FitzhughDiffusionAux) = ℝ{2}(0.0, P.σ)
    dependsOnParams(::FitzhughDiffusionAux) = (5,)
elseif parametrisation == :complexConjug
    B(t, P::FitzhughDiffusionAux) = @SMatrix [0.0  1.0;
                                (P.ϵ-P.γ-3.0*P.ϵ*P.v[1]^2) (P.ϵ-1.0-3.0*P.ϵ*P.v[1]^2)]
    β(t, P::FitzhughDiffusionAux) = ℝ{2}(0.0, 2*P.ϵ*P.v[1]^3+P.s-P.β)#check later
    σ(t, P::FitzhughDiffusionAux) = ℝ{2}(0.0, P.σ)
    dependsOnParams(::FitzhughDiffusionAux) = (1, 2, 3, 4, 5)

    function B(t, P::FitzhughDiffusionAux{T,SArray{Tuple{2},Float64,1,2}}) where T
        @SMatrix [0.0  1.0;
                  (P.ϵ-P.γ-3.0*P.ϵ*P.v[1]^2-6*P.ϵ*P.v[1]*P.v[2]) (P.ϵ-1.0-3.0*P.ϵ*P.v[1]^2)]
    end
    function β(t, P::FitzhughDiffusionAux{T,SArray{Tuple{2},Float64,1,2}}) where T
        ℝ{2}(0.0, 2*P.ϵ*P.v[1]^3+P.s-P.β+6*P.ϵ*P.v[1]^2*P.v[2])#check later
    end
end

constdiff(::FitzhughDiffusionAux) = true
b(t, x, P::FitzhughDiffusionAux) = B(t,P) * x + β(t,P)
a(t, P::FitzhughDiffusionAux) = σ(t,P) * σ(t, P)'

"""
    clone(P::FitzhughDiffusionAux, θ)

Clone the object `P`, but use a different vector of parameters `θ`.
"""
clone(P::FitzhughDiffusionAux, θ) = FitzhughDiffusionAux(θ..., P.t,
                                                         P.u, P.T, P.v)
# should copy starting point or sth, currently restricted by the same type of u and v
clone(P::FitzhughDiffusionAux, θ, v) = FitzhughDiffusionAux(θ..., P.t,
                                                         zero(v), P.T, v)
params(P::FitzhughDiffusionAux) = [P.ϵ, P.s, P.γ, P.β, P.σ]
