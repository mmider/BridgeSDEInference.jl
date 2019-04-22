import Base.valtype

#const 𝕂 = ForwardDiff.Dual{Nothing,Float64,1}

"""
    ODESolverType

Types inheriting from abstract type `ODESolverType` declare the numerical
schemes used for finding solutions to backward ODEs
"""
abstract type ODESolverType end


#TODO implement Jeffrey's priors
"""
    ImproperPrior

Flat prior
"""
struct ImproperPrior end
logpdf(::ImproperPrior, θ) = 0.0


valtype(::Val{T}) where T = T

"""
    idx(::Val{T}) where {T}

Return a tuple containing indices of parameters selected by Val{T}

# Examples
```julia-repl
julia> idx(Val((true, false, false, true, true)))
(1, 4, 5)
```
"""
function idx(::Val{T}) where T
    tuple((i for i in 1:length(T) if T[i])...)
end


"""
    moveToProperPlace(ϑ, θ, ::Val{T}) where {T}

Update parameter vector `θ` at indices specified by `Val{T}` with the
values collected in `ϑ`.

# Examples
```julia-repl
julia> moveToProperPlace([10,20,30], [1,2,3,4,5],
                         Val((true, false, true, false, true)))
5-element Array{Float64,1}:
 10.0
  2.0
 20.0
  4.0
 30.0
```
"""
function moveToProperPlace(ϑ, θ, ::Val{T}) where {T}
    v = valtype(Val{T}())
    m = length(v)
    θᵒ = zero(θ)
    idxNew = [i for i in 1:m if v[i]]
    idxOld = [i for i in 1:m if !v[i]]
    θᵒ[idxNew] .= ϑ
    θᵒ[idxOld] .= θ[idxOld]
    θᵒ
end
