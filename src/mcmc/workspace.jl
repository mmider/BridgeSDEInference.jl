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

"""
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
    accpt_rate(at::AccptTracker, ::ParamUpdate)

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


"""
    ParamHistory{T}

Stores information about parameter history
"""
mutable struct ParamHistory{T}
    θ_chain::Vector{T}  # parameter history
    counter::Int64      # index of the most recently accepted parameter vector

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
struct ActionTracker{T,S,R}
    save_iter::T    # Save the path every ... iteration
    verb_iter::S   # Print progress message to console every ... iteration
    warm_up::R      # Number of steps of the chain in which no param update is made
    param_updt::Bool    # Flag for whether to update parameters at all

    """
        ActionTracker(setup::MCMCSetup)

    Initialise tracker of what to do on a given iteration according to a `setup`
    of the mcmc sampler
    """
    function ActionTracker(setup::MCMCSetup)
        i1, i2, i3 = setup.save_iter, setup.verb_iter, setup.warm_up
        s1, s2, s3 = typeof(i1), typeof(i2), typeof(i3)
        @assert (s1 <: Number) && (s2 <: Number) && (s3 <: Number)
        new{s1,s2,s3}(i1, i2, i3, setup.param_updt)
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

"""
    Workspace{ObsScheme,S,TX,TW,R,ST}

The main container of the `mcmc` function from `mcmc.jl` in which most data
pertinent to sampling is stored
"""
struct Workspace{ObsScheme,B,S,TX,TW,R,ST}# ,Q, where Q = eltype(result)
    Wnr::Wiener{S}         # Wiener, driving law
    XXᵒ::Vector{TX}        # Diffusion proposal paths
    XX::Vector{TX}         # Accepted diffusion paths
    WWᵒ::Vector{TW}        # Driving noise of proposal
    WW::Vector{TW}         # Driving noise of the accepted paths
    Pᵒ::Vector{R}          # Guided proposals parameterised by proposal param
    P::Vector{R}           # Guided proposals parameterised by accepted param
    fpt::Vector            # Additional information about first passage times
    #TODO use vector instead for blocking
    ρ::Float64             # Memory parameter of the precond Crank-Nicolson scheme
    recompute_ODEs::Vector{Bool}    # Info on whether to recompute H,Hν,c after resp. param updt
    accpt_tracker::AccptTracker     # Object for tracking acceptance rate
    θ_chain::ParamHistory           # Object for tracking parameter history
    action_tracker::ActionTracker   # Object for tracking steps to perform on a given iteration
    skip_for_save::Int64            # Thining parameter for saving path
    paths::Vector                   # Storage with historical, accepted paths
    time::Vector{Float64}           # Storage with time axis
    blocking::B
    blidx::Int64
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
        ρ, updt_coord = deepcopy(setup.ρ), deepcopy(setup.updt_coord)
        TW, TX, S, R = eltype(WW), eltype(XX), valtype(Wnr), eltype(P)
        ST = typeof(setup.solver)

        m = length(P)
        # forcedSolve defines type by starting point, make sure it matches
        y = eltype(eltype(XX))(start_pt(x0_prior))
        for i in 1:m
            WW[i] = Bridge.samplepath(P[i].tt, zero(S))
            sample!(WW[i], Wnr)
            WW[i], XX[i] = forcedSolve(Euler(), y, WW[i], P[i])    # this will enforce adherence to domain
            while !checkFpt(ObsScheme(), XX[i], fpt[i])
                sample!(WW[i], Wnr)
                forcedSolve!(Euler(), XX[i], y, WW[i], P[i])    # this will enforce adherence to domain
            end
            y = XX[i].yy[end]
        end
        y = start_pt(x0_prior)
        ll = logpdf(x0_prior, y)
        ll += path_log_likhd(ObsScheme(), XX, P, 1:m, fpt, skipFPT=true)
        ll += lobslikelihood(P[1], y)

        XXᵒ = deepcopy(XX)
        WWᵒ = deepcopy(WW)
        Pᵒ = deepcopy(P)
        # needed for proper initialisation of the Crank-Nicolson scheme
        x0_prior = inv_start_pt(y, x0_prior, P[1])

        #TODO come back to gradient initialisation
        skip = setup.skip_for_save
        _time = collect(Iterators.flatten(p.tt[1:skip:end-1] for p in P))

        θ_history = ParamHistory(setup)

        blocking = set_blocking(setup.blocking, setup.blocking_params, P)
        display(blocking)
        B = typeof(blocking)

        (workspace = new{ObsScheme,B,S,TX,TW,R,ST}(Wnr, XXᵒ, XX, WWᵒ, WW, Pᵒ, P,
                                                   fpt, ρ, check_if_recompute_ODEs(setup),
                                                   AccptTracker(setup), θ_history,
                                                   ActionTracker(setup), skip,
                                                   [], _time, blocking, 1),
         ll = ll, x0_prior = x0_prior, θ = last(θ_history))
    end

    """
        Workspace(ws::Workspace{ObsScheme,S,TX,TW,R}, new_ρ::Float64)

    Copy constructor of `workspace`. Keeps everything the same as passed `ws`
    with the exception of new memory parameter for the preconditioned
    Crank-Nicolson scheme, which is changed to `new_ρ`.
    """
    function Workspace(ws::Workspace{ObsScheme,B,S,TX,TW,R,ST}, new_ρ::Float64
                       ) where {ObsScheme,B,S,TX,TW,R,ST}
        new{ObsScheme,B,S,TX,TW,R,ST}(ws.Wnr, ws.XXᵒ, ws.XX, ws.WWᵒ, ws.WW,
                                      ws.Pᵒ, ws.P, ws.fpt, new_ρ,
                                      ws.recompute_ODEs, ws.accpt_tracker,
                                      ws.θ_chain, ws.action_tracker,
                                      ws.skip_for_save, ws.paths, ws.time,
                                      ws.blocking, ws.blidx)
    end

    function Workspace(ws::Workspace{ObsScheme,B,S,TX,TW,R̃,ST}, P::Vector{R},
                       Pᵒ::Vector{R}, idx) where {ObsScheme,B,S,TX,TW,R̃,ST,R}
        new{ObsScheme,B,S,TX,TW,R,ST}(ws.Wnr, ws.XXᵒ, ws.XX, ws.WWᵒ, ws.WW,
                                      Pᵒ, P, ws.fpt, ws.ρ,
                                      ws.recompute_ODEs, ws.accpt_tracker,
                                      ws.θ_chain, ws.action_tracker,
                                      ws.skip_for_save, ws.paths, ws.time,
                                      ws.blocking, idx)
    end
end

eltype(::SamplePath{T}) where T = T
eltype(::Type{SamplePath{T}}) where T = T
solver_type(::Workspace{O,B,S,TX,TW,R,ST}) where {O,B,S,TX,TW,R,ST} = ST


next_set_of_blocks(ws::Workspace{O,NoBlocking}) where O = ws

"""
    next(𝔅::ChequeredBlocking, XX, θ)

Switch the set of blocks that are being updated. `XX` is the most recently
sampled (accepted) path. `θ` can be used to change parametrisation.
"""
function next_set_of_blocks(ws::Workspace{O,ChequeredBlocking}) where O
    XX, P, Pᵒ, 𝔅 = ws.XX, ws.P, ws.Pᵒ, ws.blocking
    idx = (ws.blidx % 2) + 1
    θ = params(P[1].Target)

    vs = find_end_pts(𝔅, XX, idx)
    Ls = 𝔅.Ls[idx]
    Σs = 𝔅.Σs[idx]
    ch_pts = 𝔅.change_pts[idx]

    P_new = [GuidPropBridge(P[i], Ls[i], vs[i], Σs[i], ch_pts[i], θ)
             for (i,_) in enumerate(P)]
    Pᵒ_new = [GuidPropBridge(Pᵒ[i], Ls[i], vs[i], Σs[i], ch_pts[i], θ)
              for (i,_) in enumerate(Pᵒ)]
    Workspace(ws, P_new, Pᵒ_new, idx)
end


"""
    act(action, ws::Workspace, i)

Determine whether to perform `action` on a given iteration, indexed `i`
"""
act(action, ws::Workspace, i) = act(action, ws.action_tracker, i)


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


"""
    ParamUpdtDefn

For a given, single parameter update step defines transition kernels, priors,
which coordinates are updated etc.
"""
struct ParamUpdtDefn{R,S,ST,T,U}
    updt_type::R         # The type of update (Metropolis-Hastings/conjugate etc)
    updt_coord::S        # Which coordinates to update
    t_kernel::T          # Transition kernel for a given parameter update
    priors::U            # Prior over updated parameters
    recompute_ODEs::Bool # Whether given param updt calls for recomputing H,Hν,c

    """
        ParamUpdtDefn(updt_type::R, updt_coord::S, t_kernel::T, priors::U,
                      recompute_ODEs::Bool, ::ST)

    Initialisation of the complete definition of the parameter update step
    """
    function ParamUpdtDefn(updt_type::R, updt_coord::S, t_kernel::T, priors::U,
                           recompute_ODEs::Bool, ::ST
                           ) where {R<:ParamUpdateType,S,T,U,ST<:ODESolverType}
        new{R,S,ST,T,U}(updt_type, updt_coord, t_kernel, priors, recompute_ODEs)
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
        solver = setup.solver
        recompute_ODEs = check_if_recompute_ODEs(setup)

        updates = [ParamUpdtDefn(ut, uc, tk, pr, ro, solver) for
                   (ut, uc, tk, pr, ro) in zip(setup.updt_type, setup.updt_coord,
                                               setup.t_kernel, setup.priors,
                                               recompute_ODEs)]
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
function update!(adpt::Adaptation{Val{false}}, ws::Workspace{ObsScheme}, yPr, i,
                 ll, solver::ODESolverType) where ObsScheme
    adpt, ws, yPr, ll
end


"""
    update!(adpt::Adaptation{Val{false}}, ws::Workspace{ObsScheme}, yPr, i, ll,
            solver::ODESolverType)

Update the proposal law according to the adaptive scheme and the recently saved
history of the imputed paths
"""
function update!(adpt::Adaptation{Val{true}}, ws::Workspace{ObsScheme}, yPr, i,
                 ll, solver::ODESolverType) where ObsScheme
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
            ws = Workspace(ws, adpt.ρs[adpt.N[1]])

            solveBackRec!(NoBlocking(), ws.P, ST())
            #solveBackRec!(NoBlocking(), ws.Pᵒ, ST())
            y = ws.XX[1].yy[1]
            yPr = inv_start_pt(y, yPr, ws.P[1])

            for j in 1:m
                inv_solve!(Euler(), ws.XX[j], ws.WW[j], ws.P[j])
            end
            ll = logpdf(yPr, y)
            ll += path_log_likhd(ObsScheme(), ws.XX, ws.P, 1:m, ws.fpt)
            ll += lobslikelihood(ws.P[1], y)
            adpt.N[2] = 1
            adpt.N[1] += 1
        else
            adpt.N[2] += 1
        end
    end
    adpt, ws, yPr, ll
end
