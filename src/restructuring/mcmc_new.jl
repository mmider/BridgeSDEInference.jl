#=
    MCMC smapler is defined by a sequence of block updates and a master plan
    with a schedule on how to choose and modify those blocks
=#

#===============================================================================
                        Trackers of acceptance probability
===============================================================================#

mutable struct AccptTracker{S}
    accpt::S
    prop::S
    AccptTracker(::S) where {S <: Integer} = new{S}(zero(S), zero(S))
end

function reset!(at::AccptTracker{S}) where S
    at.accpt = zero(S)
    at.prop = zero(S)
end

function acceptance_rate(at::AccptTracker{S}) where S
    at.prop == zero(S) && return 0.0
    at.accpt/at.prop
end

function register_accpt!(at::AccptTracker{S}, accepted) where S
    at.prop += one(S)
    at.accpt += one(S)*accepted
end

function register_accpt!(at::Vector{AccptTracker{S}}, accepted::Vector) where S
    @assert length(at) == length(accepted)
    for i in 1:length(at)
        register_accpt!(at[i], accepted[i])
    end
end

"""
    display(at::AccptTracker)

Show the acceptance rates
"""
function display(at::AccptTracker)
    print("Acceptance rate: ", round(accpetance_rate(at), digits=3), ".\n")
end

#===============================================================================
                        Definition of a single Gibbs step
===============================================================================#

abstract type AuxiliaryInfo end

struct UpdtAuxiliary{ST} <: AuxiliaryInfo
    solver::ST
    recompute_ODEs::Bool
    function UpdtAuxiliary(solver::ST, recompute_ODEs::Bool) where ST<:ODESolverType
        new{ST}(solver, recompute_ODEs)
    end
end


"""

Needs to have a prior, transition kernel, acceptance rate tracker, information
about the which coordinates are being updated and an object with auxiliary,
use-specific information say whether to recompute_ODEs
"""
struct MCMCParamUpdate{UpdtType,S,T,U,V} <: MCMCUpdate
    updt_coord::S
    t_kernel::T
    priors::U
    accpt_tracker::AccptTracker
    accpt_history::Vector{Bool}
    aux::V
    readjust_param::ReadjustT


    function MCMCUpdate(::UpdtType, updt_coord, θ, t_kernel::T, priors::U,
                        aux::V, readjust_param=(100, 0.1, -999, 999, 0.234, 50)
                        ) where {UpdtType<:ParamUpdateType,T,U,V<:AuxiliaryInfo}
        updt_coord = reformat_updt_coord(updt_coord, θ)
        S = typeof(updt_coord)
        new{UpdtType,S,T,U,V}(updt_coord, t_kernel, priors, AccptTracker(0),
                              Bool[], aux, named_readjust(readjust_param))
    end
end

"""
    reformat_updt_coord(updt_coord::Nothing, θ)

Chosen not to update parameters, returned object is not important
"""
reformat_updt_coord(updt_coord::Nothing, θ) = (Val((true,)),)


IntContainer = Union{Number,NTuple{N,<:Integer},Vector{<:Integer}} where N
"""
    reformat_updt_coord(updt_coord::S, θ) where S<:IntContainer

Single joint update of multiple parameters at once
"""
function reformat_updt_coord(updt_coord::S, θ) where S<:IntContainer
    @assert all([1 <= uc <= length(θ) for uc in updt_coord])
    Val{Tuple([i in updt_coord for i in 1:length(θ)])}()
end

"""
    reformat_updt_coord(updt_coord::Nothing, θ)

If the user does not use indices of coordinates to be updated it is assumed that
appropriate Val{(...)}() object is passed and nothing is done, use at your own risk
"""
reformat_updt_coord(updt_coord, θ) = updt_coord


function readjust!(pu::MCMCParamUpdate, corr_mat, mcmc_iter)
    at = pu.accpt_tracker
    p = pu.readjust_param
    readjust!(pu.t_kernel, at, p, corr_mat, mcmc_iter)
    reset!(at)
end

aux_params(::Any, aux) = aux

function register_accpt!(pu::MCMCParamUpdate, acc)
    register_accpt!(pu.accpt_tracker, acc)
    push!(accpt_history, acc)
end
#===============================================================================
                        Definition of an imputation step
===============================================================================#
struct MCMCImputation{B,ST,T} <: MCMCUpdate
    accpt_tracker::Vector{AccptTracker}
    accpt_history::Vector{T}
    ρs::Vector{Float64}
    blocking::B
    solver::ST
    readjust_param::ReadjustT

    function MCMCImputation(blocking::B, ρ::S, solver::ST=Ralston3(),
                            readjust_param=(100, 0.1, 0.00001, 0.99999, 0.234, 50)
                            ) where {B,S<:Number,ST<:ODESolverType}
        accpt_tracker = [AccptTracker(0) for _ in 1:length(blocking)]
        accpt_history = NTuple{length(blocking),Bool}[]
        T = typeof(accpt_history)
        readjust_param = named_readjust(readjust_param)
        ρs = fill(ρ, length(blocking))
        new{B,ST,T}(accpt_tracker, accpt_history, ρs, blocking, solver,
                    readjust_param)
    end
end

function readjust!(pu::MCMCImputation, ::Any, mcmc_iter)
    at = pu.accpt_tracker
    p = pu.readjust_param
    δ = compute_δ(p, mcmc_iter)
    ar = [acceptance_rate(tracker) for tracker in at]
    for i in 1:length(pu.ρs)
        pu.ρs[i] = compute_ϵ(ws.ρ[i], p, ar[i], δ, -1.0, logit, sigmoid)
    end
    display_new_ρ(ar, pu.ρs)
    reset!(at)
end

function display_new_ρ(accpt_rates, ρs)
    print("imputation acceptance rate: ")
    for i in 1:length(accpt_rates) print(round(accpt_rates[i], digits=2), "  | ") end
    print("\nnew ρs: ")
    for i in 1:length(ρs) print(round(ρs[i], digits=3), "  | ") end
    print("\n")
end

aux_params(pu::MCMCImputation, aux) = pu.blocking

function register_accpt!(pu::MCMCImputation, acc)
    register_accpt!(pu.accpt_tracker, acc)
    push!(pu.accpt_history, tuple(acc...))
end

#===============================================================================
                        Overall schedule defining MCMC
===============================================================================#

struct MCMCSchedule{T}
    num_mcmc_steps::Int64
    updt_idx::Vector{Vector{Int64}}
    start::Int64
    actions::T

    function MCMCSchedule(num_mcmc_steps, updt_idx, actions)
        try
            names = (:save, :verbose, :warm_up, :readjust, :fuse)
            for name in names
                @assert name in keys(actions)
            end
        catch e
            if isa(e, AssertionError)
                @assert length(actions) == 5
                actions = (save=actions[1], verbose=actions[2], warm_up=actions[3],
                           readjust=actions[4], fuse=actions[5])

                print("`actions` has been passed without names. The names are ",
                      " set automatically:\n",
                      "save imputation every ", actions.save, " iterations,\n",
                      "print to console every ", actions.verbose, " iterations,\n",
                      "warm up the sampler for ", actions.warm_up, " iterations,\n",
                      "readjust proposals using function ", actions.readjust, "\n",
                      ",\nfuse the most correlated param updates at iterations ",
                      "determined by the function ", actions.fuse, ".\n")
            else
                rethrow(e)
            end
        end
        T = typeof(actions)
        new{T}(num_mcmc_steps, updt_idx, 1, actions)
    end
end

function Base.iterate(iter::MCMCSchedule, state=(iter.start, 1))
    element, i = state

    i > iter.num_mcmc_steps && return nothing

    at = iter.actions
    actions = (idx=iter.updt_idx[element],
               save=(i > at.warm_up && i % at.save == 0),
               verbose=(i % at.verbose == 0),
               param_updt=(at.param_updt && i > at.warm_up),
               readjust=at.readjust(i),
               fuse=at.fuse(i),
               iter=i)

    return (actions, (transition(iter, element), i + 1))
end

Base.length(iter::MCMCSchedule) = iter.length

function Base.eltype(iter::MCMCSchedule)
    NamedTuple{(:idx, :save, :verbose, :param_updt, :readjust, :fuse, :iter),
               Tuple{Array{Int64,1},Bool,Bool,Bool,Bool,Bool,Int64}}
end
transition(schedule::MCMCSchedule, elem) = mod1(elem + 1, length(schedule.updt_idx))



#===============================================================================
                                The main routine
===============================================================================#

function mcmc(setup_mcmc::MCMCSetup, schedule::MCMCSchedule, setup::T) where T <: ModelSetup
    ws, ll, θ = create_workspace(setup)
    ws_mcmc = create_workspace(setup_mcmc, schedule, θ)

    aux = nothing
    for step in schedule
        step.save && save_imputed!(ws)
        for i in step.idx
            ws = next(ws, ws_mcmc.updates[i])
            ll, acc, θ = update!(ws_mcmc.updates[i], ws, θ, ll, step, aux)
            aux = aux_params(ws_mcmc.updates[i], aux)
            update!(ws_mcmc, acc, θ, i)
            step.verbose && print("\n")
        end
        step.verbose && print("-----------------------------------------------",
                              "------\n")
        step.readjust && readjust!(ws_mcmc)
        step.fuse && fuse!(ws_mcmc, schedule)
    end
    display_summary(ws, ws_mcmc)
    ws
end





foo = MCMCSchedule(10, [[1,2,3],[3,4],[5,6,7,8]], (1,2,3,x->(x%5==0),true, x->(x%4==0)))


for (i,f) in enumerate(foo)
    if i == 5
        foo.updt_idx[1][1]=10000
    end
    print(f, "\n")
end
