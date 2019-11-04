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

#===============================================================================
                        Workspace for the MCMC chain
===============================================================================#

struct MCMCWorkspace{T,S}
    θ_chain::Vector{T}
    updates::S
    adjust_param
    function MCMCWorkspace(setup::MCMCSetup, schedule, θ::T) where T
        # TODO create an object that pre-allocates the memory based on schedule
        # this is difficult due to potential fusing of kernels which results
        # in a random overall number of updates
        θ_chain = [θ]
        S = typeof(setup.updates)
        new{T,S}(θ_chain, setup.updates)
    end
end


function update!(ws::MCMCWorkspace, acc, θ, i)
    typeof(ws.updates[i]) != Imputation && push!(ws.θ_chain, θ)
    register_accpt!(ws.updates[i], acc)
end

function readjust!(ws::MCMCWorkspace, mcmc_iter)
    for i in 1:length(ws.updates)
        # `nothing` will be a correlation matrix
        readjust!(ws.updates[i], nothing, mcmc_iter)
    end
end

function fuse!(ws::MCMCWorkspace, schedule)
    corr_mat = find_correlation(ws.θ_chain)
    idx1, idx2 = find_fuse_indices(corr_mat, ws.updates)
    # fuse
end


"""
    getindex(g::GibbsDefn, i::Int)

Return `i`th definition of parameter update
"""
getindex(g::MCMCWorkspace, i::Int) = g.updates[i]

"""
    length(g::GibbsDefn{N})

Return the total number of parameter updates in a single Gibbs sweep
"""
length(g::MCMCWorkspace) = length(g.updates)


mutable struct SingleElem{T} val::T end

set!(x::SingleElem{T}, y::T) where T = (x.val = y)

#===============================================================================
                    Workspace for the Diffusion Model
===============================================================================#
"""
    Workspace{ObsScheme,S,TX,TW,R,ST}

The main container of the `mcmc` function from `mcmc.jl` in which most data
pertinent to sampling is stored
"""
struct Workspace{ObsScheme,S,TX,TW,R,TP,TZ}# ,Q, where Q = eltype(result)
    # Related to imputed path
    Wnr::Wiener{S}         # Wiener, driving law
    XXᵒ::Vector{TX}        # Diffusion proposal paths
    XX::Vector{TX}         # Accepted diffusion paths
    WWᵒ::Vector{TW}        # Driving noise of proposal
    WW::Vector{TW}         # Driving noise of the accepted paths
    Pᵒ::Vector{R}          # Guided proposals parameterised by proposal param
    P::Vector{R}           # Guided proposals parameterised by accepted param
    fpt::Vector            # Additional information about first passage times
    # Related to historically sampled paths
    skip_for_save::Int64            # Thining parameter for saving path
    paths::Vector                   # Storage with historical, accepted paths
    time::Vector{Float64}           # Storage with time axis
    # Related to the starting point
    x0_prior::TP
    z::SingleElem{TZ}
    #recompute_ODEs::Vector{Bool}    # Info on whether to recompute H,Hν,c after resp. param updt

    """
        Workspace(setup::MCMCSetup{ObsScheme})

    Initialise workspace of the mcmc sampler according to a `setup` variable
    """
    function Workspace(setup::DiffusionSetup{ObsScheme}) where ObsScheme
        x0_prior, Wnr = deepcopy(setup.x0_prior), deepcopy(setup.Wnr)
        XX, WW = deepcopy(setup.XX), deepcopy(setup.WW)
        P, fpt = deepcopy(setup.P), deepcopy(setup.fpt)

        # forcedSolve defines type by the starting point, make sure it matches
        x0_guess = eltype(eltype(XX))(setup.x0_guess)
        TW, TX, S, R = eltype(WW), eltype(XX), valtype(Wnr), eltype(P)
        TP = typeof(x0_prior)
        m = length(P)

        y = copy(x0_guess)
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
        y = x0_guess
        ll = ( logpdf(x0_prior, y)
               + path_log_likhd(ObsScheme(), XX, P, 1:m, fpt, skipFPT=true)
               + lobslikelihood(P[1], y) )

        XXᵒ, WWᵒ, Pᵒ = deepcopy(XX), deepcopy(WW), deepcopy(P)

        # compute the white noise that generates x0_guess under the initial posterior
        z = inv_start_pt(y, x0_prior, P[1])
        TZ = typeof(z)
        z = SingleElem{TZ}(z)

        skip = setup.skip_for_save
        _time = collect(Iterators.flatten(p.tt[1:skip:end-1] for p in P))

         #check_if_recompute_ODEs(setup)
         (workspace = new{ObsScheme,S,TX,TW,R,TP,TZ}(Wnr, XXᵒ, XX, WWᵒ, WW, Pᵒ,
                                                     P, fpt, skip, [], _time,
                                                     x0_prior, z),
          ll = ll, θ = params(P[1].Target))
    end

    function Workspace(ws::Workspace{ObsScheme,S,TX,TW,R̃,TP,TZ}, P::Vector{R},
                       Pᵒ::Vector{R}) where {ObsScheme,S,TX,TW,R̃,R,TP,TZ}
        new{ObsScheme,S,TX,TW,R,TP,TZ}(ws.Wnr, ws.XXᵒ, ws.XX, ws.WWᵒ, ws.WW, Pᵒ,
                                       P, ws.fpt, ws.skip_for_save, ws.paths,
                                       ws.time, ws.x0_prior, ws.z)
    end
end

eltype(::SamplePath{T}) where T = T
eltype(::Type{SamplePath{T}}) where T = T
solver_type(::Workspace{O,B,ST}) where {O,B,ST} = ST


next(ws::Workspace, ::Any) = ws
function next(ws::Workspace, updt::Imputation{<:Block})
    XX, P, Pᵒ, bl = ws.XX, ws.P, ws.Pᵒ, updt.blocking
    θ = params(P[1].Target)
    vs = find_end_pts(bl, XX)

    P_new = [GuidPropBridge(Pi, bl.Ls[i], vs[i], bl.Σs[i], bl.change_pts[i],
                            θ, bl.aux_flags[i]) for (i,Pi) in enumerate(P)]
    Pᵒ_new = [GuidPropBridge(Pᵒi, bl.Ls[i], vs[i], bl.Σs[i], bl.change_pts[i],
                             θ, bl.aux_flags[i]) for (i,Pᵒi) in enumerate(Pᵒ)]
    Workspace(ws, P_new, Pᵒ_new)
end

#prepare_mem_param(ρ::Number, ::NoBlocking) = [[ρ]]

#function prepare_mem_param(ρ::Number, blocking::ChequeredBlocking)
#    [[ρ for _ in block_seq] for block_seq in blocking.accpt_tracker.accpt]
#end


#NOTE deprecated
#act(action::Readjust, ws::Workspace, i) = (typeof(ws.blocking) <:ChequeredBlocking) && act(action, ws.action_tracker, i)


"""
    savePath!(ws, wsXX, bXX)

Save the entire path spanning all segments in `XX`. Only 1 in every `ws.skip`
points is saved to reduce storage space. To-be-saved `XX` is set to `wsXX` or
`bXX` depending on whether blocking is used.
"""
function save_imputed!(ws::Workspace)
    skip = ws.skip_for_save
    push!(ws.paths, collect(Iterators.flatten(ws.XX[i].yy[1:skip:end-1]
                                               for i in 1:length(ws.XX))))
end






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
                inv_solve!(Euler(), ws.XX[j], ws.WW[j], ws.P[j])
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


function create_workspace(setup::MCMCSetup, schedule::MCMCSchedule, θ)
    MCMCWorkspace(setup, schedule, θ)
end

function create_workspace(setup::T) where {T <: ModelSetup}
    Workspace(setup)
end
