using Bridge
using StaticArrays
import Bridge: b, σ, B, β, a, constdiff
const ℝ = SVector{N,T} where {N,T}
import Base.display

"""
    FitzhughDiffusion <: ContinuousTimeProcess{ℝ{2}}

Struct defining FitzHugh-Nagumo model
"""
struct FitzhughDiffusion{T,TP} <: ContinuousTimeProcess{ℝ{2,T}}
    param::TP
    ϵ::T
    s::T
    γ::T
    β::T
    σ::T

    function FitzhughDiffusion(ϵ::T, s::T, γ::T, β::T, σ::T) where T
        TP = Val{:regular}
        new{T,TP}(TP(), ϵ, s, γ, β, σ)
    end

    function FitzhughDiffusion(::Val{S}, ϵ::T, s::T, γ::T, β::T, σ::T
                               ) where {T,S}
        TP = Val{S}
        new{T,TP}(TP(), ϵ, s, γ, β, σ)
    end

    function FitzhughDiffusion(sym::Symbol, ϵ::T, s::T, γ::T, β::T, σ::T
                               ) where T
        checkParamValid(sym)
        TP = Val{sym}
        new{T,TP}(TP(), ϵ, s, γ, β, σ)
    end

    function FitzhughDiffusion(param::String, ϵ::T, s::T, γ::T, β::T, σ::T
                               ) where T
        TP = stringToParam(param)
        new{T,TP}(TP(), ϵ, s, γ, β, σ)
    end
end

function checkParamValid(s::Symbol)
    sym = [:regular, :simpleAlter, :complexAlter, :simpleConjug, :complexConjug]
    (s in sym) || error("Invalid parametrisation of the FitzHugh-Nagumo model")
end

function stringToParam(s::String)
    s = lowercase(s)
    if s == "regular"
        param = Val{:regular}
    elseif s in ["simplealter", "simple alter", "simple alternative"]
        param = Val{:simpleAlter}
    elseif s in ["complexalter", "complex alter", "complex alternative"]
        param = Val{:complexAlter}
    elseif s in ["simpleconjug", "simple conjug", "simple conjugate"]
        param = Val{:simpleConjug}
    elseif s in ["complexconjug", "complex conjug", "complex conjugate"]
        param = Val{:complexConjug}
    else
        error("Invalid parametrisation of the FitzHugh-Nagumo model")
    end
    param
end

# REGULAR PARAMETRISATION
# -----------------------

function b(t, x, P::FitzhughDiffusion{T,Val{:regular}}) where T
    ℝ{2}((x[1]-x[2]-x[1]^3+P.s)/P.ϵ, P.γ*x[1]-x[2] + P.β)
end

function σ(t, x, P::FitzhughDiffusion{T,Val{:regular}}) where T
    ℝ{2}(0.0, P.σ)
end


# ALTERNATIVE PARAMETRISATION
# ---------------------------

function b(t, x, P::FitzhughDiffusion{T,TP}
           ) where {T,TP <: Union{Val{:simpleAlter},Val{:complexAlter}}}
    ℝ{2}(x[2], -( (P.γ-1.0)*x[1] + x[1]^3 + P.ϵ*x[2] - P.s + P.β
                  + (3.0*x[1]^2 - 1.0)*x[2])/P.ϵ )
end

function σ(t, x, P::FitzhughDiffusion{T,TP}
           ) where {T,TP <: Union{Val{:simpleAlter},Val{:complexAlter}}}
    ℝ{2}(0.0, P.σ/P.ϵ)
end


# CONJUGATE PARAMETRISATION
# -------------------------

function b(t, x, P::FitzhughDiffusion{T,TP}
           ) where {T,TP <: Union{Val{:simpleConjug},Val{:complexConjug}}}
    ℝ{2}(x[2], ((P.ϵ - P.γ)*x[1] - P.ϵ*(x[1]^3 + (3.0*x[1]^2 - 1.0)*x[2])
                + P.s - P.β - x[2]))
end

function σ(t, x, P::FitzhughDiffusion{T,TP}
           ) where {T,TP <: Union{Val{:simpleConjug},Val{:complexConjug}}}
    ℝ{2}(0.0, P.σ)
end

# APPLICABLE TO ALL PARAMETRISATIONS
# ----------------------------------

constdiff(::FitzhughDiffusion) = true
clone(P::FitzhughDiffusion, θ) = FitzhughDiffusion(P.param, θ...)
params(P::FitzhughDiffusion) = [P.ϵ, P.s, P.γ, P.β, P.σ]


# Basis functions for conjugate update
phi(::Val{0}, t, x, P::FitzhughDiffusion) = -x[2]
phi(::Val{1}, t, x, P::FitzhughDiffusion) = x[1]-x[1]^3+(1-3*x[1]^2)*x[2]
phi(::Val{2}, t, x, P::FitzhughDiffusion) = one(x[1])
phi(::Val{3}, t, x, P::FitzhughDiffusion) = -x[1]
phi(::Val{4}, t, x, P::FitzhughDiffusion) = zero(x[1])
phi(::Val{5}, t, x, P::FitzhughDiffusion) = zero(x[1])



"""
    struct FitzhughDiffusionAux <: ContinuousTimeProcess{ℝ{2}}

Struct defining proposal diffusion (proposal for sampling from FitzHugh-Nagumo
diffusion)
"""
struct FitzhughDiffusionAux{R,S1,S2,TP} <: ContinuousTimeProcess{ℝ{2,R}}
    param::TP
    ϵ::R
    s::R
    γ::R
    β::R
    σ::R
    t::Float64
    u::S1
    T::Float64
    v::S2

    function FitzhughDiffusionAux(ϵ::R, s::R, γ::R, β::R, σ::R, t::Float64,
                                  u::S1, T::Float64, v::S2) where {R,S1,S2}
        TP = Val{:regular}
        new{R,S1,S2,TP}(TP(), ϵ, s, γ, β, σ, t, u, T, v)
    end

    function FitzhughDiffusionAux(::Val{S}, ϵ::R, s::R, γ::R, β::R, σ::R,
                                  t::Float64, u::S1, T::Float64, v::S2
                                  ) where {R,S,S1,S2}
        TP = Val{S}
        new{R,S1,S2,TP}(TP(), ϵ, s, γ, β, σ, t, u, T, v)
    end

    function FitzhughDiffusionAux(sym::Symbol, ϵ::R, s::R, γ::R, β::R, σ::R,
                                  t::Float64, u::S1, T::Float64, v::S2
                                  ) where {R,S1,S2}
        checkParamValid(sym)
        TP = Val{sym}
        new{R,S1,S2,TP}(TP(), ϵ, s, γ, β, σ, t, u, T, v)
    end

    function FitzhughDiffusionAux(param::String, ϵ::R, s::R, γ::R, β::R, σ::R,
                                  t::Float64, u::S1, T::Float64, v::S2
                                  ) where {R,S1,S2}
        TP = stringToParam(param)
        new{R,S1,S2,TP}(TP(), ϵ, s, γ, β, σ, t, u, T, v)
    end
end

"""
    depends_on_params(::FitzhughDiffusionAux)

Declare which parameters (1=>`ϵ`, 2=>`s`, 3=>`γ`, 4=>`β`, 5=>`σ`) the
auxiliary diffusion depends upon. Used for finding out which parameter
update requires also updating the values of the grid of `H`'s and `r`'s.
"""
function depends_on_params end


# REGULAR PARAMETRISATION
# -----------------------

function B(t, P::FitzhughDiffusionAux{T,S1,S2,Val{:regular}}) where {T,S1,S2}
    @SMatrix [1/P.ϵ-3*P.v^2/P.ϵ  -1/P.ϵ; P.γ -1.0]
end

function β(t, P::FitzhughDiffusionAux{T,S1,S2,Val{:regular}}) where {T,S1,S2}
    ℝ{2}(P.s/P.ϵ+2*P.v^3/P.ϵ, P.β)
end

function σ(t, P::FitzhughDiffusionAux{T,S1,S2,Val{:regular}}) where {T,S1,S2}
    ℝ{2}(0.0, P.σ)
end

function depends_on_params(::FitzhughDiffusionAux{T,S1,S2,Val{:regular}}) where {T,S1,S2}
    (1, 2, 3, 4, 5)
end


# SIMPLE ALTERNATIVE PARAMETRISATION
# ----------------------------------

function B(t, P::FitzhughDiffusionAux{T,S1,S2,Val{:simpleAlter}}) where {T,S1,S2}
    @SMatrix [0.0  1.0; 0.0 0.0]
end

function β(t, P::FitzhughDiffusionAux{T,S1,S2,Val{:simpleAlter}}) where {T,S1,S2}
    ℝ{2}(0.0, 0.0)
end

function σ(t, P::FitzhughDiffusionAux{T,S1,S2,Val{:simpleAlter}}) where {T,S1,S2}
    ℝ{2}(0.0, P.σ/P.ϵ)
end

function depends_on_params(::FitzhughDiffusionAux{T,S1,S2,Val{:simpleAlter}}) where {T,S1,S2}
    (1, 5)
end

# COMPLEX ALTERNATIVE PARAMETRISATION
# -----------------------------------

function B(t, P::FitzhughDiffusionAux{T,S1,S2,Val{:complexAlter}}) where {T,S1,S2}
    @SMatrix [0.0  1.0; (1.0-P.γ-3.0*P.v[1]^2)/P.ϵ (1.0-P.ϵ-3.0*P.v[1]^2)/P.ϵ]
end

function β(t, P::FitzhughDiffusionAux{T,S1,S2,Val{:complexAlter}}) where {T,S1,S2}
    ℝ{2}(0.0, (2*P.v[1]^3+P.s-P.β)/P.ϵ)
end

function σ(t, P::FitzhughDiffusionAux{T,S1,S2,Val{:complexAlter}}) where {T,S1,S2}
    ℝ{2}(0.0, P.σ/P.ϵ)
end

function depends_on_params(::FitzhughDiffusionAux{T,S1,S2,Val{:complexAlter}}) where {T,S1,S2}
    (1, 2, 3, 4, 5)
end

function B(t, P::FitzhughDiffusionAux{T,S1,SArray{Tuple{2},Float64,1,2},Val{:complexAlter}}) where {T,S1}
    @SMatrix [0.0  1.0;
              (1.0-P.γ-3.0*P.v[1]^2-6*P.v[1]*P.v[2])/P.ϵ (1.0-P.ϵ-3.0*P.v[1]^2)/P.ϵ]
end

function β(t, P::FitzhughDiffusionAux{T,S1,SArray{Tuple{2},Float64,1,2},Val{:complexAlter}}) where {T,S1}
    ℝ{2}(0.0, (2*P.v[1]^3+P.s-P.β+6*P.v[1]^2*P.v[2])/P.ϵ)#check later
end


# SIMPLE CONJUGATE PARAMETRISATION
# --------------------------------

function B(t, P::FitzhughDiffusionAux{T,S1,S2,Val{:simpleConjug}}) where {T,S1,S2}
    @SMatrix [0.0  1.0; 0.0 0.0]
end

function β(t, P::FitzhughDiffusionAux{T,S1,S2,Val{:simpleConjug}}) where {T,S1,S2}
    ℝ{2}(0.0, 0.0)
end

function σ(t, P::FitzhughDiffusionAux{T,S1,S2,Val{:simpleConjug}}) where {T,S1,S2}
    ℝ{2}(0.0, P.σ)
end

function depends_on_params(::FitzhughDiffusionAux{T,S1,S2,Val{:simpleConjug}}) where {T,S1,S2}
    (5,)
end


# COMPLEX CONJUGATE PARAMETRISATION
# ---------------------------------
function B(t, P::FitzhughDiffusionAux{T,S1,S2,Val{:complexConjug}}) where {T,S1,S2}
    @SMatrix [0.0  1.0; (P.ϵ-P.γ-3.0*P.ϵ*P.v[1]^2) (P.ϵ-1.0-3.0*P.ϵ*P.v[1]^2)]
end

function β(t, P::FitzhughDiffusionAux{T,S1,S2,Val{:complexConjug}}) where {T,S1,S2}
    ℝ{2}(0.0, 2*P.ϵ*P.v[1]^3+P.s-P.β) # check later
end

function σ(t, P::FitzhughDiffusionAux{T,S1,S2,Val{:complexConjug}}) where {T,S1,S2}
    ℝ{2}(0.0, P.σ)
end

function depends_on_params(::FitzhughDiffusionAux{T,S1,S2,Val{:complexConjug}}) where {T,S1,S2}
    (1, 2, 3, 4, 5)
end

function B(t, P::FitzhughDiffusionAux{T,S1,SArray{Tuple{2},Float64,1,2},Val{:complexConjug}}) where {T,S1}
    @SMatrix [0.0  1.0;
              (P.ϵ-P.γ-3.0*P.ϵ*P.v[1]^2-6*P.ϵ*P.v[1]*P.v[2]) (P.ϵ-1.0-3.0*P.ϵ*P.v[1]^2)]
end

function β(t, P::FitzhughDiffusionAux{T,S1,SArray{Tuple{2},Float64,1,2},Val{:complexConjug}}) where {T,S1}
    ℝ{2}(0.0, 2*P.ϵ*P.v[1]^3+P.s-P.β+6*P.ϵ*P.v[1]^2*P.v[2]) # check later
end


# APPLICABLE TO ALL PARAMETRISATIONS
# ----------------------------------

constdiff(::FitzhughDiffusionAux) = true
b(t, x, P::FitzhughDiffusionAux) = B(t,P) * x + β(t,P)
a(t, P::FitzhughDiffusionAux) = σ(t,P) * σ(t, P)'

"""
    clone(P::FitzhughDiffusionAux, θ)

Clone the object `P`, but use a different vector of parameters `θ`.
"""
clone(P::FitzhughDiffusionAux, θ) = FitzhughDiffusionAux(P.param, θ..., P.t,
                                                         P.u, P.T, P.v)
# should copy starting point or sth, currently restricted by the same type of u and v
clone(P::FitzhughDiffusionAux, θ, v) = FitzhughDiffusionAux(P.param, θ..., P.t,
                                                            zero(v), P.T, v)
params(P::FitzhughDiffusionAux) = (P.ϵ, P.s, P.γ, P.β, P.σ)
param_names(::FitzhughDiffusion) = (:ϵ, :s, :γ, :β, :σ)

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

function orig_params(P::FitzhughDiffusion{<:Any,<:Val{:complexConjug}})
    ϵ = 1/P.ϵ
    ϵ, P.s*ϵ, P.γ*ϵ, P.β*ϵ, P.σ*ϵ
end

_MSG = ["\n---------------------------------------------------\n",
        "The target diffusion model is set to FitzHugh-Nagumo diffusion, a solution ",
        " to an SDE:\n",
        "dYₜ = (Yₜ-Yₜ³-Xₜ+s)/ϵ dt,\ndXₜ = (γYₜ-Xₜ+β) dt + σ dWₜ,  t∈[0,T].\n",
        "dYₜ = Ẏₜ dt,\ndẎₜ = [(1-γ)Yₜ-Yₜ³-ϵẎₜ+s-β+(1-3Yₜ²)Ẏₜ] dt + σ/ϵ dWₜ,  t∈[0,T].\n",
        "dYₜ = Ẏₜ dt,\ndẎₜ = [(ϵ-γ)Yₜ-ϵYₜ³-Ẏₜ+s-β+ϵ(1-3Yₜ²)Ẏₜ] dt + σ dWₜ,  t∈[0,T].\n",
        "The auxiliary diffusion is set to ",
        ", a solution to an SDE:\n",
        "dỸₜ = [(1-3y²)Ỹₜ-X̃ₜ+s+2y³] dt,\ndX̃ₜ = (γỸₜ-X̃ₜ+β) dt + σ dWₜ,  t∈[0,T],\n",
        "dIₜ = Bₜ dt,\ndBₜ = σ/ϵ dWₜ,  t∈[0,T],\n",
        "dỸₜ = X̃ₜ dt,\ndX̃ₜ = [(1-γ-3y²-6yẏ)Ỹₜ+(1-ϵ-3y²)X̃ₜ+(2y³+s-β+6y²ẏ)]/ϵ dt + σ/ϵ dWₜ,  t∈[0,T],\n",
        "dIₜ = Bₜ dt,\ndBₜ = σ dWₜ,  t∈[0,T],\n",
        "dỸₜ = X̃ₜ dt,\ndX̃ₜ = {[ϵ(1-3y²-6yẏ)-γ]Ỹₜ+[ϵ(1-3y²)-1]X̃ₜ+[ϵ(2y³+6y²ẏ)+s-β]} dt + σ dWₜ,  t∈[0,T],\n",
        "where (y,ẏ) denotes an end-point of (Y,Ẏ).\n",
        "---------------------------------------------------\n",
        ]

function display(::FitzhughDiffusion{T,Val{:regular}}) where T
    print(_MSG[1], _MSG[2], "(Y,X)", _MSG[3], _MSG[4], _MSG[15])
end

function display(::FitzhughDiffusion{T,Val{:simpleAlter}}) where T
    print(_MSG[1], _MSG[2], "(Y,Ẏ)", _MSG[3], _MSG[5], _MSG[15])
end

function display(::FitzhughDiffusion{T,Val{:complexAlter}}) where T
    print(_MSG[1], _MSG[2], "(Y,Ẏ)", _MSG[3], _MSG[5], _MSG[15])
end

function display(::FitzhughDiffusion{T,Val{:simpleConjug}}) where T
    print(_MSG[1], _MSG[2], "(Y,Ẏ)", _MSG[3], _MSG[6], _MSG[15])
end

function display(::FitzhughDiffusion{T,Val{:complexConjug}}) where T
    print(_MSG[1], _MSG[2], "(Y,Ẏ)", _MSG[3], _MSG[6], _MSG[15])
end

function display(::FitzhughDiffusionAux{T,S1,S2,Val{:regular}}) where {T,S1,S2}
    print(_MSG[1], _MSG[7], "(Ỹ,X̃)", _MSG[8], _MSG[9], _MSG[14], _MSG[15])
end

function display(::FitzhughDiffusionAux{T,S1,S2,Val{:simpleAlter}}) where {T,S1,S2}
    print(_MSG[1], _MSG[7], "(I,B)", _MSG[8], _MSG[10], _MSG[15])
end

function display(::FitzhughDiffusionAux{T,S1,S2,Val{:complexAlter}}) where {T,S1,S2}
    print(_MSG[1], _MSG[7], "(Ỹ,X̃)", _MSG[8], _MSG[11], _MSG[14], _MSG[15])
end

function display(::FitzhughDiffusionAux{T,S1,S2,Val{:simpleConjug}}) where {T,S1,S2}
    print(_MSG[1], _MSG[7], "(I,B)", _MSG[8], _MSG[12], _MSG[15])
end

function display(::FitzhughDiffusionAux{T,S1,S2,Val{:complexConjug}}) where {T,S1,S2}
    print(_MSG[1], _MSG[7], "(Ỹ,X̃)", _MSG[8], _MSG[13], _MSG[14], _MSG[15])
end
