#=
    ---------------------------------------------------------------------------
    Implements a few collections that the main mcmc sampler from the file
    `mcmc.jl` uses to keep track of the current state and history of where it
    was. The following structures are defined:
    - AccptTracker   : for tracking historical acceptance rate
    - ParamHistory   : for storing the parameter chain
    - ActionTracker  : keeps track of what to do on a given iteration
    - Workspace      : main workspace for `mcmc` function from `mcmc.jl`
    - ParamUpdtDefn  : defines a single parameter update step
    - GibbsDefn      : defines an entire gibbs sweep of parameter updates
    ---------------------------------------------------------------------------
=#

import Base: last, getindex, length, display, eltype

"""
    AccptTracker

Tracks historical acceptance rate of the chain
"""
mutable struct AccptTracker
    accpt_imp::Int64            # a number of accepted imputation samples
    prop_imp::Int64             # a number of proposed imputation samples
    accpt_updt::Vector{Int64}   # a number of accepted parameter updates (one per parameter)
    prop_updt::Vector{Int64}    # a number of proposed parameter updates (one per parameter)
    updt_len::Int64             # total number of parameter update steps

    """
        AccptTracker(setup::MCMCSetup)

    Initialise tracker of acceptance rate according to a `setup` of the mcmc
    sampler
    """
    function AccptTracker(setup::MCMCSetup)
        updt_len = length(setup.updt_coord)
        accpt_imp = 0
        prop_imp = 0
        accpt_updt = [0 for i in 1:updt_len]
        prop_updt = [0 for i in 1:updt_len]
        new(accpt_imp, prop_imp, accpt_updt, prop_updt, updt_len)
    end
end

"""const
    update!(at::AccptTracker, ::ParamUpdate, i, accepted::Bool)

Update acceptance tracker: increment parameter update info on the `i`th
coordinate with 1 proposed and---if `accepted` is true---then also 1 accepted
sample
"""
function update!(at::AccptTracker, ::ParamUpdate, i, accepted::Bool)
    at.prop_updt[i] += 1
    at.accpt_updt[i] += 1*accepted
end

"""
    update!(at::AccptTracker, ::Imputation, accepted::Bool)

Update acceptance tracker: increment imputation info with 1 proposed and---if
`accepted` is true---then also 1 accepted sample
"""
function update!(at::AccptTracker, ::Imputation, accepted::Bool)
    at.prop_imp += 1
    at.accpt_imp += 1*accepted
end

"""
    accpt_rate(at::AccptTracker, ::ParamUpdate)const

Return current acceptance rate for parameter updates
"""
accpt_rate(at::AccptTracker, ::ParamUpdate) = at.accpt_updt./at.prop_updt

"""
    accpt_rate(at::AccptTracker, ::Imputation)

Return current acceptance rate for imputation
"""
accpt_rate(at::AccptTracker, ::Imputation) = at.accpt_imp/at.prop_imp

"""
    display(at::AccptTracker)

Show the acceptance rates
"""
function display(at::AccptTracker)
    print("Imputation acceptance rate: ", accpt_rate(at, Imputation()),
          ".\nParameter update acceptance rate: ",
          accpt_rate(at, ParamUpdate()), ".\n")
end

function reset!(at::AccptTracker, ::ParamUpdate)
    for i in length(at.accpt_updt)
        at.accpt_updt[i] = 0
        at.prop_updt[i] = 0
    end
end

function reset!(at::AccptTracker, ::Imputation)
    at.accpt_imp = 0
    at.prop_imp = 0
end

"""
    ParamHistory{T}

Stores information about parameter history
"""
mutable struct ParamHistory{T}
    θ_chain::Vector{T}  # parameter history
    counter::Int64      # index of the most recentlyconst  accepted parameter vector

    """
        ParamHistory(setup::MCMCSetup)

    Initialise tracker of parameter history according to a `setup` of the mcmc
    sampler
    """
    function ParamHistory(setup::MCMCSetup)
        N, n = setup.num_mcmc_steps, setup.warm_up
        updt_len = length(setup.updt_coord)

        θ = params(setup.P˟)
        T = typeof(θ)
        θ_chain = Vector{T}(undef, (N-n)*updt_len+1)
        θ_chain[1] = copy(θ)
        new{T}(θ_chain, 1)
    end
end

"""
    update!(ph::ParamHistory, θ)

Update parameter history with a new accepted sample θ
"""
function update!(ph::ParamHistory, θ)
    ph.counter += 1
    ph.θ_chain[ph.counter] = copy(θ)
end

"""
    last(ph::ParamHistory)

Return a copy of the most recently accepted paramter vector
"""
last(ph::ParamHistory) = copy(ph.θ_chain[ph.counter])


"""
    ActionTracker

Keeps track of what to do on a given iteration
"""
struct ActionTracker{T,S,R,U}
    save_iter::T    # Save the path every ... iteration
    verb_iter::S   # Print progress message to console every ... iteration
    warm_up::R      # Number of steps of the chain in which no param update is made
    readjust::U
    param_updt::Bool    # Flag for whether to update parameters at all

    """
        ActionTracker(setup::MCMCSetup)

    Initialise tracker of what to do on a given iteration according to a `setup`
    of the mcmc sampler
    """
    function ActionTracker(setup::MCMCSetup)
        i1, i2, i3 = setup.save_iter, setup.verb_iter, setup.warm_up
        i4 = setup.pCN_readjust_param.step
        s1, s2, s3, s4 = typeof(i1), typeof(i2), typeof(i3), typeof(i4)
        @assert (s1 <: Number) && (s2 <: Number) && (s3 <: Number) && (s4 <: Number)
        new{s1,s2,s3,s4}(i1, i2, i3, i4, setup.param_updt)
    end
end

"""
    act(::SavePath, at::ActionTracker, i)

Determine whether to save path on a given iteration, indexed `i`
"""
function act(::SavePath, at::ActionTracker, i)
    (i > at.warm_up) && (i % at.save_iter == 0)
end

"""
    act(::Verbose, at::ActionTracker, i)

Determine whether to print out information to a console on a given iteration,
indexed `i`
"""
act(::Verbose, at::ActionTracker, i) = (i % at.verb_iter == 0)

"""
    act(::ParamUpdate, at::ActionTracker, i)

Determine whether to update parameters on a given iteration, indexed `i`
"""
act(::ParamUpdate, at::ActionTracker, i) = at.param_updt && (i > at.warm_up)

act(::Readjust, at::ActionTracker, i) = (i % at.readjust == 0) && (i > at.warm_up)


mutable struct SingleElem{T} val::T end

set!(x::SingleElem{T}, y::T) where T = (x.val = y)

const RhoInfoType = NamedTuple{(:step, :scale, :minδ, :maxρ, :trgt, :offset),
                               Tuple{Int64, Float64, Float64, Float64, Float64, Int64}}
"""
    Workspace{ObsScheme,S,TX,TW,R,ST}

The main container of the `mcmc` function from `mcmc.jl` in which most data
pertinent to sampling is stored
"""
struct Workspace{ObsScheme,B,ST,S,TX,TW,R,TP,TZ,Tθ}# ,Q, where Q = eltype(result)
    Wnr::Wiener{S}         # Wiener, driving law
    XXᵒ::Vector{TX}        # Diffusion proposal paths
    XX::Vector{TX}         # Accepted diffusion paths
    WWᵒ::Vector{TW}        # Driving noise of proposal
    WW::Vector{TW}         # Driving noise of the accepted paths
    Pᵒ::Vector{R}          # Guided proposals parameterised by proposal param
    P::Vector{R}           # Guided proposals parameterised by accepted param
    fpt::Vector            # Additional information about first passage times
    ρ::Vector{Vector{Float64}}      # Memory parameter of the precond Crank-Nicolson scheme
    recompute_ODEs::Vector{Bool}    # Info on whether to recompute H,Hν,c after resp. param updt
    accpt_tracker::AccptTracker     # Object for tracking acceptance rate
    accpt_tracker_short::AccptTracker
    θ_chain::ParamHistory           # Object for tracking parameter history
    action_tracker::ActionTracker   # Object for tracking steps to perform on a given iteration
    skip_for_save::Int64            # Thining parameter for saving path
    paths::Vector                   # Storage with historical, accepted paths
    time::Vector{Float64}           # Storage with time axis
    blocking::B
    blidx::Int64
    x0_prior::TP
    z::SingleElem{TZ}
    pCN_readjust_param::RhoInfoType
    θ_readjust_param::Tθ
    #result::Vector{Q} #TODO come back to later
    #resultᵒ::Vector{Q} #TODO come back to later

    """
        Workspace(setup::MCMCSetup{ObsScheme})

    Initialise workspace of the mcmc sampler according to a `setup` variable
    """
    function Workspace(setup::MCMCSetup{ObsScheme}) where ObsScheme
        # just to make sure that nothing gets messed up if the user decides
        # to later modify `setup` use deepcopies
        x0_prior, Wnr = deepcopy(setup.x0_prior), deepcopy(setup.Wnr)
        XX, WW = deepcopy(setup.XX), deepcopy(setup.WW)
        P, fpt = deepcopy(setup.P), deepcopy(setup.fpt)
        updt_coord = deepcopy(setup.updt_coord)
        pCN_readjust = deepcopy(setup.pCN_readjust_param)
        θ_readjust = deepcopy(setup.θ_readjust_param)

        # forcedSolve defines type by the starting point, make sure it matches
        x0_guess = eltype(eltype(XX))(setup.x0_guess)
        TW, TX, S, R = eltype(WW), eltype(XX), valtype(Wnr), eltype(P)
        ST, TP, Tθ = typeof(setup.solver), typeof(x0_prior), typeof(θ_readjust)


        m = length(P)

        y = copy(x0_guess)
        for i in 1:m
            WW[i] = Bridge.samplepath(P[i].tt, zero(S))
            sample!(WW[i], Wnr)
            WW[i], XX[i] = forcedSolve(EulerMaruyamaBounded(), y, WW[i], P[i])    # this will enforce adherence to domain
            while !checkFpt(ObsScheme(), XX[i], fpt[i])
                sample!(WW[i], Wnr)
                forcedSolve!(EulerMaruyamaBounded(), XX[i], y, WW[i], P[i])    # this will enforce adherence to domain
            end
            y = XX[i].yy[end]
        end
        y = x0_guess
        ll = logpdf(x0_prior, y)
        ll += path_log_likhd(ObsScheme(), XX, P, 1:m, fpt, skipFPT=true)
        ll += lobslikelihood(P[1], y)

        XXᵒ = deepcopy(XX)
        WWᵒ = deepcopy(WW)
        Pᵒ = deepcopy(P)

        # compute the white noise that generates x0_guess under the initial posterior
        z = inv_start_pt(y, x0_prior, P[1])
        TZ = typeof(z)
        z = SingleElem{TZ}(z)

        #TODO come back to gradient initialisation
        skip = setup.skip_for_save
        _time = collect(Iterators.flatten(p.tt[1:skip:end-1] for p in P))

        θ_history = ParamHistory(setup)

        blocking = set_blocking(setup.blocking, setup.blocking_params, P)
        ρ = prepare_mem_param(setup.ρ, blocking)
        display(blocking)
        B = typeof(blocking)

        (workspace = new{ObsScheme,B,ST,S,TX,TW,R,TP,TZ,Tθ}(Wnr, XXᵒ, XX, WWᵒ, WW,
                                                         Pᵒ, P, fpt, ρ,
                                                         check_if_recompute_ODEs(setup),
                                                         AccptTracker(setup),
                                                         AccptTracker(setup),
                                                         θ_history,
                                                         ActionTracker(setup),
                                                         skip, [], _time,
                                                         blocking, 1, x0_prior,
                                                         z, pCN_readjust,
                                                         θ_readjust),
         ll = ll, θ = last(θ_history))
    end

    # NOTE this constructor is no longer in use, can be removed
    """
        Workspace(ws::Workspace{ObsScheme,S,TX,TW,R}, new_ρ::Float64)

    Copy constructor of `workspace`. Keeps everything the same as passed `ws`
    with the exception of new memory parameter for the preconditioned
    Crank-Nicolson scheme, which is changed to `new_ρ`.
    """
    function Workspace(ws::Workspace{ObsScheme,B,ST,S,TX,TW,R,TP,TZ,Tθ}, new_ρ::Vector{Vector{Float64}}
                       ) where {ObsScheme,B,ST,S,TX,TW,R,TP,TZ,Tθ}
        new{ObsScheme,B,ST,S,TX,TW,R,TP,TZ,Tθ}(ws.Wnr, ws.XXᵒ, ws.XX, ws.WWᵒ,
                                            ws.WW, ws.Pᵒ, ws.P, ws.fpt, new_ρ,
                                            ws.recompute_ODEs, ws.accpt_tracker,
                                            ws.accpt_tracker_short,
                                            ws.θ_chain, ws.action_tracker,
                                            ws.skip_for_save, ws.paths, ws.time,
                                            ws.blocking, ws.blidx, ws.x0_prior,
                                            ws.z, ws.pCN_readjust_param,
                                            ws.θ_readjust_param)
    end

    function Workspace(ws::Workspace{ObsScheme,B,ST,S,TX,TW,R̃,TP,TZ,Tθ},
                       P::Vector{R}, Pᵒ::Vector{R}, idx
                       ) where {ObsScheme,B,ST,S,TX,TW,R̃,R,TP,TZ,Tθ}
        new{ObsScheme,B,ST,S,TX,TW,R,TP,TZ,Tθ}(ws.Wnr, ws.XXᵒ, ws.XX, ws.WWᵒ,
                                            ws.WW, Pᵒ, P, ws.fpt, ws.ρ,
                                            ws.recompute_ODEs, ws.accpt_tracker,
                                            ws.accpt_tracker_short,
                                            ws.θ_chain, ws.action_tracker,
                                            ws.skip_for_save, ws.paths, ws.time,
                                            ws.blocking, idx, ws.x0_prior, ws.z,
                                            ws.pCN_readjust_param,
                                            ws.θ_readjust_param)
    end
end

eltype(::SamplePath{T}) where T = T
eltype(::Type{SamplePath{T}}) where T = T
solver_type(::Workspace{O,B,ST}) where {O,B,ST} = ST


next_set_of_blocks(ws::Workspace{O,NoBlocking}) where O = ws

"""
    next(𝔅::ChequeredBlocking, XX, θ)

Switch the set of blocks that are being updated. `XX` is the most recently
sampled (accepted) path. `θ` can be used to change parametrisation.
"""
function next_set_of_blocks(ws::Workspace{O,<:ChequeredBlocking}) where O
    XX, P, Pᵒ, 𝔅 = ws.XX, ws.P, ws.Pᵒ, ws.blocking
    idx = (ws.blidx % 2) + 1
    θ = params(P[1].Target)

    vs = find_end_pts(𝔅, XX, idx)
    Ls = 𝔅.Ls[idx]
    Σs = 𝔅.Σs[idx]
    ch_pts = 𝔅.change_pts[idx]
    aux_flags = 𝔅.aux_flags[idx]

    P_new = [GuidPropBridge(P[i], Ls[i], vs[i], Σs[i], ch_pts[i], θ, aux_flags[i])
             for (i,_) in enumerate(P)]
    Pᵒ_new = [GuidPropBridge(Pᵒ[i], Ls[i], vs[i], Σs[i], ch_pts[i], θ, aux_flags[i])
              for (i,_) in enumerate(Pᵒ)]
    Workspace(ws, P_new, Pᵒ_new, idx)
end

prepare_mem_param(ρ::Number, ::NoBlocking) = [[ρ]]

function prepare_mem_param(ρ::Number, blocking::ChequeredBlocking)
    [[ρ for _ in block_seq] for block_seq in blocking.accpt_tracker.accpt]
end



"""
    act(action, ws::Workspace, i)

Determine whether to perform `action` on a given iteration, indexed `i`
"""
act(action, ws::Workspace, i) = act(action, ws.action_tracker, i)

#NOTE deprecated
#act(action::Readjust, ws::Workspace, i) = (typeof(ws.blocking) <:ChequeredBlocking) && act(action, ws.action_tracker, i)


"""
    savePath!(ws, wsXX, bXX)

Save the entire path spanning all segments in `XX`. Only 1 in every `ws.skip`
points is saved to reduce storage space. To-be-saved `XX` is set to `wsXX` or
`bXX` depending on whether blocking is used.
"""
function save_path!(ws)
    skip = ws.skip_for_save
    push!(ws.paths, collect(Iterators.flatten(ws.XX[i].yy[1:skip:end-1]
                                               for i in 1:length(ws.XX))))
end

sigmoid(x, a=1.0) = 1.0 / (1.0 + exp(-a*x))
logit(x, a=1.0) = (log(x) - log(1-x))/a

function readjust_pCN!(ws, mcmc_iter)
    at = ws.blocking.short_term_accpt_tracker
    p = ws.pCN_readjust_param
    δ = max(p.minδ, p.scale/sqrt(max(1.0, mcmc_iter/p.step-p.offset)))
    accpt_rates = acceptance(at)
    for (i, a_i) in enumerate(accpt_rates)
        for (j, a_ij) in enumerate(a_i)
            ws.ρ[i][j] = min(sigmoid(logit(ws.ρ[i][j]) - (2*(a_ij > p.trgt)-1)*δ), p.maxρ)
        end
    end
    display_acceptance_rate(ws.blocking, true)
    print_pCN(ws)
    reset!(at)
end

function readjust_pCN!(ws::Workspace{OS,NoBlocking}, mcmc_iter) where OS
    at = ws.accpt_tracker_short
    p = ws.pCN_readjust_param
    δ = max(p.minδ, p.scale/sqrt(max(1.0, mcmc_iter/p.step-p.offset)))
    a_r = accpt_rate(at, Imputation())
    ws.ρ[1][1] = min(sigmoid(logit(ws.ρ[1][1]) - (2*(a_r > p.trgt)-1)*δ), p.maxρ)
    print("imputation acceptance rate: ", a_r, "\n")
    print("new ρ: ", ws.ρ[[1]], "\n")
    reset!(at, Imputation())
end

function readjust_tk(ws, mcmc_iter, param_updt_defn)
    at = ws.accpt_tracker_short
    p = ws.θ_readjust_param
    δ = max(p.minδ, p.scale/sqrt(max(1.0, mcmc_iter/p.step-p.offset)))
    a_r = accpt_rate(at, ParamUpdate())
    fns = [λ->max(min(λ + (2*(a_r[i] > p.trgt)-1)*δ, p.maxδ), p.minδ) for i in 1:length(a_r)]
    print("param updt acceptance rate: ", a_r, "\n")
    gd = GibbsDefn(param_updt_defn, fns, p.idx_MH, p.idx_θ)
    print("new transition kernels: \n")
    for i in 1:length(gd)
        print(i, ". ", gd.updates[i].t_kernel, "\n")
    end
    gd
end


#NOTE _print_info defined in `blocking_schedule`
function print_pCN(ws)
    print("\nρ parameter:\n----------------------\n")
    _print_info(ws.ρ[1])
    _print_info(ws.ρ[2])
end


"""
    ParamUpdtDefn

For a given, single parameter update step defines transition kernels, priors,
which coordinates are updated etc.
"""
struct ParamUpdtDefn{R,S,T,U}
    updt_type::R         # The type of update (Metropolis-Hastings/conjugate etc)
    updt_coord::S        # Which coordinates to update
    t_kernel::T          # Transition kernel for a given parameter update
    priors::U            # Prior over updated parameters
    recompute_ODEs::Bool # Whether given param updt calls for recomputing H,Hν,c

    """
        ParamUpdtDefn(updt_type::R, updt_coord::S, t_kernel::T, priors::U,
                      recompute_ODEs::Bool)

    Initialisation of the complete definition of the parameter update step
    """
    function ParamUpdtDefn(updt_type::R, updt_coord::S, t_kernel::T, priors::U,
                           recompute_ODEs::Bool
                           ) where {R<:ParamUpdateType,S,T,U}
        new{R,S,T,U}(updt_type, updt_coord, t_kernel, priors, recompute_ODEs)
    end
end

"""
    GibbsDefn

Definition of the entire Gibbs sweep.
"""
struct GibbsDefn{N}
    updates::NTuple{N,ParamUpdtDefn}

    """
        GibbsDefn(setup)

    Initialises Gibbs sweep according to the `setup`
    """
    function GibbsDefn(setup)
        recompute_ODEs = check_if_recompute_ODEs(setup)

        updates = [ParamUpdtDefn(ut, uc, tk, pr, ro) for (ut, uc, tk, pr, ro)
                   in zip(setup.updt_type, setup.updt_coord, setup.t_kernel,
                          setup.priors, recompute_ODEs)]
        new{length(updates)}(Tuple(updates))
    end

    function GibbsDefn(gd::GibbsDefn, fns, idx_MH, idx_θ)
        t_kernels = [(i in idx_MH ?
                      new_tkernel(gd.updates[i].t_kernel, fns[i], idx_θ[i]) :
                      gd.updates[i].t_kernel)
                     for i in 1:length(gd.updates)]
        updates = [ParamUpdtDefn(u.updt_type, u.updt_coord, t_kernels[i],
                                 u.priors, u.recompute_ODEs)
                   for (i, u) in enumerate(gd.updates)]
        new{length(updates)}(Tuple(updates))
    end
end

"""
    getindex(g::GibbsDefn, i::Int)

Return `i`th definition of parameter update
"""
getindex(g::GibbsDefn, i::Int) = g.updates[i]

"""
    length(g::GibbsDefn{N})

Return the total number of parameter updates in a single Gibbs sweep
"""
length(g::GibbsDefn{N}) where N = N




"""
    init_adaptation!(adpt::Adaptation{Val{false}}, ws::Workspace)

Nothing to do when no adaptation needs to be done
"""
init_adaptation!(adpt::Adaptation{Val{false}}, ws::Workspace) = nothing

"""
    init_adaptation!(adpt::Adaptation{Val{true}}, ws::Workspace)

Resize internal container with paths in `adpt` to match the length of imputed
paths
"""
function init_adaptation!(adpt::Adaptation{Val{true}}, ws::Workspace)
    m = length(ws.XX)
    resize!(adpt, m, [length(ws.XX[i]) for i in 1:m])
end


"""
    update!(adpt::Adaptation{Val{false}}, ws::Workspace{ObsScheme}, yPr, i, ll,
            solver::ODESolverType)

Nothing to be done for no adaptation
"""
function update!(adpt::Adaptation{Val{false}}, ws::Workspace{ObsScheme}, i,
                 ll) where ObsScheme
    adpt, ll
end


"""
    update!(adpt::Adaptation{Val{false}}, ws::Workspace{ObsScheme}, yPr, i, ll,
            solver::ODESolverType)

Update the proposal law according to the adaptive scheme and the recently saved
history of the imputed paths
"""
function update!(adpt::Adaptation{Val{true}}, ws::Workspace{ObsScheme,B,ST},
                 i, ll) where {ObsScheme,B,ST}
    if i % adpt.skip == 0
        if adpt.N[2] == adpt.sizes[adpt.N[1]]
            X_bar = mean_trajectory(adpt)
            m = length(ws.P)
            for j in 1:m
                Pt = recentre(ws.P[j].Pt, ws.XX[j].tt, X_bar[j])
                update_λ!(Pt, adpt.λs[adpt.N[1]])
                ws.P[j] = GuidPropBridge(ws.P[j], Pt)

                Ptᵒ = recentre(ws.Pᵒ[j].Pt, ws.XX[j].tt, X_bar[j])
                update_λ!(Ptᵒ, adpt.λs[adpt.N[1]])
                ws.Pᵒ[j] = GuidPropBridge(ws.Pᵒ[j], Ptᵒ)
            end

            solve_back_rec!(NoBlocking(), ws, ws.P)
            #solveBackRec!(NoBlocking(), ws.Pᵒ, ST())
            y = ws.XX[1].yy[1]
            z = inv_start_pt(y, ws.x0_prior, ws.P[1])
            set!(ws.z, z)

            for j in 1:m
                inv_solve!(EulerMaruyamaBounded(), ws.XX[j], ws.WW[j], ws.P[j])
            end
            ll = logpdf(ws.x0_prior, y)
            ll += path_log_likhd(ObsScheme(), ws.XX, ws.P, 1:m, ws.fpt)
            ll += lobslikelihood(ws.P[1], y)
            adpt.N[2] = 1
            adpt.N[1] += 1
        else
            adpt.N[2] += 1
        end
    end
    adpt, ll
end
