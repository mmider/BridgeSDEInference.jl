#=
    MCMC smapler is defined by a sequence of block updates and a master plan
    with a schedule on how to choose and modify those blocks
=#

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

function register_accpt!(at, accepted)
    at.prop += 1
    at.accpt += 1*accepted
end


abstract type MCMCUpdate end

"""

Needs to have a prior, transition kernel, acceptance rate tracker, information
about the which coordinates are being updated and an object with auxiliary,
use-specific information say whether to recompute_ODEs
"""
struct MCMCParamUpdate{R,S,T,U,V} <: MCMCUpdate
    updt_type::R
    updt_coord::S
    t_kernel::T
    priors::U
    accpt_tracker::AccptTracker
    readjust_param::V
    aux::Z

    function MCMCUpdate(updt_type::R, updt_coord::S, t_kernel::T, priors::U,
                        readjust_param::V, aux::Z) where {R<:ParamUpdateType,S,T,U,V,Z}
        accpt_tracker = AccptTracker(0)
        new{R,S,T,U,V,Z}(updt_type, updt_coord, t_kernel, priors, accpt_tracker,
                         readjust_param, aux)
    end
end

function readjust!(pu::MCMCParamUpdate, corr_mat, mcmc_iter)
    at = pu.accpt_tracker
    p = pu.readjust_param
    readjust!(pu.t_kernel, at, p, corr_mat, mcmc_iter)
    reset!(at)
end


struct MCMCImputation{B,T} <: MCMCUpdate
    accpt_tracker::Vector{AccptTracker}
    ρs::Vector{Float64}
    blocking::B
    readjust_param::T

    function MCMCImputation(___TODO_blocking, ρ, readjust_param)
        accpt_tracker = [AccptTracker(0) for _ in 1:num_blocks]
        T = typeof(readjust_param)
        @assert typeof(ρ) <: Number
        ρs = fill(ρ, num_blocks)
        new{T}(accpt_tracker, ρs, readjust_param)
    end
end

sigmoid(x, a=1.0) = 1.0 / (1.0 + exp(-a*x))
logit(x, a=1.0) = (log(x) - log(1-x))/a

function readjust!(pu::MCMCImputation, ::Any, mcmc_iter)
    at = pu.accpt_tracker
    p = pu.readjust_param
    δ = max(p.minδ, p.scale/sqrt(max(1.0, mcmc_iter/p.step-p.offset)))
    ar = [acceptance_rate(tracker) for tracker in at]
    for i in 1:length(pu.ρs)
        pu.ρs[i] = min(sigmoid(logit(ws.ρ[i]) - (2*(ar[i] > p.trgt)-1)*δ), p.maxρ)
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


struct MCMCSchedule{T}
    num_mcmc_steps::Int64
    updt_idx::Vector{Vector{Int64}}
    start::Int64
    actions::T

    function MCMCSchedule(num_mcmc_steps, updt_idx, actions)
        try
            names = (:save, :verbose, :warm_up, :readjust, :param_updt, :fuse)
            for name in names
                @assert name in keys(actions)
            end
        catch e
            if isa(e, AssertionError)
                @assert length(actions) == 6
                actions = (save=actions[1], verbose=actions[2], warm_up=actions[3],
                           readjust=actions[4], param_updt=actions[5],
                           fuse=actions[6])

                print("`actions` has been passed without names. The names are ",
                      " set automatically:\n",
                      "save imputation every ", actions.save, " iterations,\n",
                      "print to console every ", actions.verbose, " iterations,\n",
                      "warm up the sampler for ", actions.warm_up, " iterations,\n",
                      "readjust proposals using function ", actions.readjust, "\n",
                      "update parameters: ", actions.param_updt,
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


function mcmc(kernels, schedule::MCMCSchedule, setup::T) where T <: ModelSetup
    ws, ll, θ = Workspace(setup, schedule)
    ws_mcmc = Workspace(schedule)

    for step in schedule
        step.save && save_imputed!(ws)
        ws = next(ws)
        for i in step.idx
            ll, θ = update!(ws_mcmc.updates[i], ws, θ, ll, step)
            update!(ws_mcmc, ws, acc, θ, i)
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
