using Bridge
import Base.valtype

abstract type DiffusionDomain end

struct UnboundedDomain <: DiffusionDomain end

boundSatisfied(::UnboundedDomain, x) = true

struct LowerBoundedDomain{T,N} <: DiffusionDomain
    bounds::NTuple{N,T}
    coords::NTuple{N,Integer}

    function LowerBoundedDomain(bounds::NTuple{N,T},
                                coords::NTuple{N,Integer}) where {N,T}
        new{T,N}(bounds, coords)
    end

    function LowerBoundedDomain(bounds::NTuple{N,T}, coords) where {N,T}
        @assert length(coords) == N
        @assert all([typeof(c) <: Integer for c in coords])
        new{T,N}(bounds, Tuple(coords))
    end

    function LowerBoundedDomain(bounds::Vector{T}, coords) where T
        N = length(bounds)
        @assert length(coords) == N
        @assert all([typeof(c) <: Integer for c in coords])
        new{T,N}(Tuple(bounds), Tuple(coords))
    end
end

function boundSatisfied(d::LowerBoundedDomain{T,N}, x) where {T,N}
    for i in 1:N
        (x[d.coords[i]] < d.bounds[i]) && return false
    end
    true
end

struct UpperBoundedDomain{T,N} <: DiffusionDomain
    bounds::NTuple{N,T}
    coords::NTuple{N,Integer}

    function UpperBoundedDomain(bounds::NTuple{N,T},
                                coords::NTuple{N,Integer})  where {T,N}
        new{T,N}(bounds, coords)
    end

    function UpperBoundedDomain(bounds::NTuple{N,T}, coords) where {T,N}
        @assert length(coords) == N
        @assert all([typeof(c) <: Integer for c in coords])
        new{T,N}(bounds, Tuple(coords))
    end

    function UpperBoundedDomain(bounds::Vector{T}, coords) where T
        N = length(bounds)
        @assert length(coords) == N
        @assert all([typeof(c) <: Integer for c in coords])
        new{T,N}(Tuple(bounds), Tuple(coords))
    end
end

function boundSatisfied(d::UpperBoundedDomain{T,N}, x) where {T,N}
    for i in 1:N
        (x[d.coords[i]] > d.bounds[i]) && return false
    end
    true
end

struct BoundedDomain{T,N1,N2} <: DiffusionDomain
    lowBds::LowerBoundedDomain{T,N1}
    upBds::UpperBoundedDomain{T,N2}

    function BoundedDomain(lowBds::LowerBoundedDomain{T,N1},
                           upBds::UpperBoundedDomain{T,N2}) where {T,N1,N2}
        new{T,N1,N2}(lowBds, upBds)
    end

    function BoundedDomain(lowBds, lowBdsCoords, upBds, upBdsCoords)
        lowBdsObj = lowerBoundedDomain(lowBds, lowBdsCoords)
        upBdsObj = upperBoundedDomain(upBds, upBdsCoords)
        T,N1 = valtype(lowBdsObj)
        S,N2 = valtype(upBdsObj)
        @assert T == S
        new{T,N1,N2}(lowBdsObj, upBdsObj)
    end
end

boundSatisfied(d, x) = boundSatisfied(d.lowBds, x) && boundSatisfied(d.upBds, x)

valtype(::LowerBoundedDomain{T,N}) where {T,N} = T,N
valtype(::UpperBoundedDomain{T,N}) where {T,N} = T,N
valtype(::BoundedDomain{T,N1,N2}) where {T,N1,N2} = T,N1,N2
DomainSomehowBounded = Union{LowerBoundedDomain,UpperBoundedDomain,BoundedDomain}

domain(::Any) = UnboundedDomain() # by default no restrictions




"""
    checkDomainAdherence(P::Vector{ContinuousTimeProcess},
                         XX::Vector{SamplePath}, iRange)

Verify whether all paths in the range `iRange`, i.e. `XX[i].yy`, i in `iRange`
fall on the interior of the domain of diffusions `P[i]`, i in `iRange`
"""
function checkDomainAdherence(P::Vector{S}, XX::Vector{T}, iRange
                              ) where {S<:ContinuousTimeProcess, T<:SamplePath}
    for i in iRange
        !checkDomainAdherence(P[i], XX[i]) && return false
    end
    true
end

"""
    checkDomainAdherence(P::ContinuousTimeProcess, XX::SamplePath,
                         d::UnboundedDomain=domain(P))

For unrestricted domains there is nothing to check
"""
function checkDomainAdherence(P::ContinuousTimeProcess, XX::SamplePath,
                              d::UnboundedDomain=domain(P.Target))
    true
end

"""
    checkDomainAdherence(P::ContinuousTimeProcess, XX::SamplePath,
                         d::DiffusionDomain=domain(P))

Verify whether path `XX.yy` falls on the interior of the domain of diffusion `P`
"""
function checkDomainAdherence(P::ContinuousTimeProcess, XX::SamplePath,
                              d::DiffusionDomain=domain(P.Target))
    N = length(XX)
    for i in 1:N
        !boundSatisfied(d, XX.yy[i]) && false
    end
    true
end
