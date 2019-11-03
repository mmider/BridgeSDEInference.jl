#NOTE this might be an overkill and using vectors with indices might be
# sufficient

# assumes throughout that T is an Ntuple of Booleans
import Base: length





"""
    reformat_updt_coord(updt_coord::Nothing, θ)

Chosen not to update parameters, returned object is not important
"""
reformat_updt_coord(updt_coord::Nothing, θ) = (Val((true,)),)


IntContainer = Union{Number,NTuple{N,<:Integer},Vector{<:Integer}} where N
IntLongContainer = Union{NTuple{N,<:Integer},Vector{<:Integer}} where N
DoubleContainer = Union{Vector{<:IntLongContainer},NTuple{N,Tuple}} where N
"""
    reformat_updt_coord(updt_coord::S, θ) where S<:IntContainer

Single joint update of multiple parameters at once
"""
function reformat_updt_coord(updt_coord::S, θ) where S<:IntContainer
    (int_to_val(updt_coord, θ),)
end

"""
    reformat_updt_coord(updt_coord::S, θ) where S<:DoubleContainer

Multiple updates, reformat from indices to update to a tuple of Val{...}()
"""
function reformat_updt_coord(updt_coord::S, θ) where S<:DoubleContainer
    Tuple([int_to_val(uc, θ) for uc in updt_coord])
end

"""
    int_to_val(updt_coord::S, θ)

Transform from a container of indices to update to a
Val{tuple of true/false one-hot-encoding}()
"""
function int_to_val(updt_coord::S, θ) where S<:IntContainer
    @assert all([1 <= uc <= length(θ) for uc in updt_coord])
    Val{Tuple([i in updt_coord for i in 1:length(θ)])}()
end


"""
    reformat_updt_coord(updt_coord::Nothing, θ)

If the user does not use indices of coordinates to be updated it is assumed that
appropriate Val{(...)}() object is passed and nothing is done, use at your own risk
"""
reformat_updt_coord(updt_coord, θ) = updt_coord




@generated function thetaex(::Val{T}, θ) where T
    z = Expr(:tuple, 1.0, (:(θ[$i]) for i in 1:length(T) if  !T[i])...)
    return z
end

@generated function thetainc(::Val{T}, θ) where T
    z = Expr(:tuple, (:(θ[$i]) for i in 1:length(T) if T[i])...)
    return z
end

@generated function indices(::Val{T}) where T
    z = Expr(:tuple, (i for i in 1:length(T) if T[i])...)
    return z
end

function matinc(mat, coord_idx)
    idx = [indices(coord_idx)...]
    n = size(mat)[1]
    mat_idx = [(n*i).+idx for i in 0:(n-1)]
    mat_idx = reshape(collect(Iterators.flatten(mat_idx)), (n,n))
    mat[mat_idx]
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
function move_to_proper_place(ϑ, θ, coord_idx::Val{T}) where T
    θᵒ = copy(θ)
    θᵒ[[indices(coord_idx)...]] = wrap(ϑ)
    θᵒ
end

function move_to_proper_place!(ϑ, θ, coord_idx::Val{T}) where T
    θ[[indices(coord_idx)...]] = wrap(ϑ)
    θ
end

wrap(x::Any) = x
wrap(x::Number) = [x]
truelength(::Val{T}) where T = sum(T)
length(::Val{T}) where T = length(T)
