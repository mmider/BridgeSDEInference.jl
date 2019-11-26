import Base.valtype
import Distributions.logpdf

"""
    ODESolverType

Types inheriting from abstract type `ODESolverType` declare the numerical
schemes used for finding solutions to backward ODEs
"""
abstract type ODESolverType end

"""
    ODEChangePt

Types inheriting from abstract type `ODEChangePt` decide upon which ODE solvers
are to be used and when to ultimately compute the triplet H,Hν,c.
"""
abstract type ODEChangePt end

"""
    NoChangePt <: ODEChangePt

Struct indicating that only solvers for H,Hν,c are to be employed. The field `λ`
indicates the amount of space (in the units of the number of elements of vector)
that needs to be nevertheless reserved for L,M⁺,μ (even though the latter are
not used).
"""
struct NoChangePt <: ODEChangePt
    λ::Int64
    NoChangePt(λ=0) = new(λ)
end

"""
    SimpleChangePt <: ODEChangePt

Struct indicating that both types of solvers are to be used; for H,Hν,c as well
as for L,M⁺,μ. The ODE solvers for L,M⁺,μ are used first on the terminal part
of the interval, where `λ` gives the lenght of this terminal interval. The ODE
solvers for H,Hν,c are used from then on.
"""
struct SimpleChangePt <: ODEChangePt
    λ::Int64
    SimpleChangePt(λ=0) = new(λ)
end

"""
    get_change_pt(changePt::ODEChangePt)

Return the length of terminal interval over which ODE solvers for L,M⁺,μ are
used
"""
get_change_pt(changePt::ODEChangePt) = changePt.λ


"""
    AbstractObsScheme

Types inheriting from abstract type `AbstractObsScheme` define the scheme
according to which a stochastic process has been observed
"""
abstract type AbstractObsScheme end


"""
    PartObs <: AbstractObsScheme

Type acting as a flag for partially observed diffusions
"""
struct PartObs <: AbstractObsScheme end


"""
    FPT <: AbstractObsScheme

Observation scheme in which only first passage times are observed
"""
struct FPT <: AbstractObsScheme end


"""
    ParamUpdateType

Types inheriting from abstract type `ParamUpdateType` define the way in which
parameters are to be updated by the MCMC sampler
"""
abstract type ParamUpdateType end

"""
    ConjugateUpdt <: ParamUpdateType

Type acting as a flag for update from full conditional (conjugate to a prior)
"""
struct ConjugateUpdt <: ParamUpdateType end

struct PseudoConjugateUpdt <: ParamUpdateType end

"""
    MetropolisHastingsUpdt <: ParamUpdateType

Flag for performing update according to Metropolis Hastings step
"""
struct MetropolisHastingsUpdt <: ParamUpdateType end

"""
    LangevinUpdt <: ParamUpdateType
Flag for performing Metropolis adjusted Langevin updates
"""
struct LangevinUpdt <: ParamUpdateType end

"""
    BlockingSchedule

Types inheriting from abstract type `BlockingSchedule` define the schedule
according to which blocking is done
"""
abstract type BlockingSchedule end

"""
    struct NoBlocking <: BlockingSchedule end

Regular updates with no blocking
"""
struct NoBlocking <: BlockingSchedule end


#TODO implement Jeffrey's priors
"""
    ImproperPrior

Flat prior
"""
struct ImproperPrior end
logpdf(::ImproperPrior, coord_idx, θ) = 0.0

struct ImproperPosPrior end
logpdf(::ImproperPosPrior, coord_idx, θ) = -sum(log.(θ[[indices(coord_idx)...]]))

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

get_aux_flag(::Any) = nothing

# NOTE quick, but potentially dangerous way to generically handle the new
# cloning functionality, change later!
clone(A, B, C, D) = clone(A, B, C)


abstract type MCMCUpdate end
abstract type ModelSetup end
