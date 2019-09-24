import Base: last, getindex, length

mutable struct AccptTracker
    accpt_imp::Int64
    prop_imp::Int64
    accpt_updt::Vector{Int64}
    prop_updt::Vector{Int64}
    updt_len::Int64

    function AccptTracker(setup::MCMCSetup)
        updt_len = length(setup.updt_coord)
        accpt_imp = 0
        prop_imp = 0
        accpt_updt = [0 for i in 1:updt_len]
        prop_updt = [0 for i in 1:updt_len]
        new(accpt_imp, prop_imp, accpt_updt, prop_updt, updt_len)
    end
end

function update!(at::AccptTracker, ::ParamUpdate, i, accepted::Bool)
    at.prop_updt[i] += 1
    at.accpt_updt[i] += 1*accepted
end

function update!(at::AccptTracker, ::Imputation, accepted::Bool)
    at.prop_imp += 1
    at.accpt_imp += 1*accepted
end

accpt_rate(at::AccptTracker, ::ParamUpdate) = at.accpt_updt./at.prop_updt
accpt_rate(at::AccptTracker, ::Imputation) = at.accpt_imp/at.prop_imp

function display(at::AccptTracker)
    print("Imputation acceptance rate: ", accpt_rate(at, Imputation()),
          ".\nParameter update acceptance rate: ",
          accpt_rate(at, ParamUpdate()), ".\n")
end

mutable struct ParamHistory{T}
    θ_chain::Vector{T}
    counter::Int64
    function ParamHistory(setup::MCMCSetup)
        N, n = setup.num_mcmc_steps, setup.warmp_up
        updt_len = length(setup.updt_coord)

        θ = params(setup.P˟)
        T = typeof(θ)
        θ_chain = Vector{T}(undef, (N-n)*updt_len+1)
        θ_chain[1] = copy(θ)
        new{T}(θ_chain, 1)
    end
end

function update!(ph::ParamHistory, θ)
    ph.counter += 1
    ph.θ_chain[ph.counter] = copy(θ)
end


function check_if_recompute_ODEs(setup::MCMCSetup)
    [any([e in depends_on_params(setup.P[1].Pt) for e
          in idx(uc)]) for uc in setup.updt_coord]
end

last(ph::ParamHistory) = copy(ph.θ_chain[ph.counter])

struct ActionTracker
    save_iter::Int64
    verb_iter::Int64
    warm_up::Int64
    param_updt::Bool

    function ActionTracker(setup::MCMCSetup)
        new(setup.save_iter, setup.verb_iter, setup.warm_up, setup.param_updt)
    end
end

function act(::SavePath, at::ActionTracker, i)
    (i > at.warm_up) && (i % at.save_iter == 0)
end

act(::Verbose, at::ActionTracker, i) = (i % at.verb_iter == 0)

act(::ParamUpdate, at::ActionTracker, i) = at.param_updt && (i > at.warm_up)

struct Workspace{ObsScheme,S,TX,TW,R}# ,Q, where Q = eltype(result)
    Wnr::Wiener{S}
    XXᵒ::Vector{TX}
    XX::Vector{TX}
    WWᵒ::Vector{TW}
    WW::Vector{TW}
    Pᵒ::Vector{R}
    P::Vector{R}
    fpt::Vector
    ρ::Float64 #TODO use vector instead for blocking
    recompute_ODEs::Vector{Bool}
    accpt_tracker::AccptTracker
    θ_chain::ParamHistory
    action_tracker::ActionTracker
    skip_for_save::Int64
    no_blocking_used::Bool #TODO deprecate this by depracating seperate containers for 𝔅
    paths::Vector
    time::Vector{Float64}
    #result::Vector{Q} #TODO come back to later
    #resultᵒ::Vector{Q} #TODO come back to later

    function Workspace(setup::MCMCSetup{ObsScheme}) where ObsScheme#, P::Vector{R}, m, yPr::StartingPtPrior{T},
                       #::S, fpt, ρ, updtCoord) where {K,R,T,S}
        x0_prior, Wnr, XX, WW = setup.x0_prior, setup.Wnr, setup.XX, setup.WW
        P, fpt, ρ, updt_coord = setup.P, setup.fpt, setup.ρ, setup.updt_coord
        TW, TX, S, R = eltype(WW), eltype(XX), valtype(Wnr), eltype(P)

        y = start_pt(x0_prior)
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
        ll += pathLogLikhd(ObsScheme(), XX, P, 1:m, fpt, skipFPT=true)
        ll += lobslikelihood(P[1], y)

        XXᵒ = deepcopy(XX)
        WWᵒ = deepcopy(WW)
        # needed for proper initialisation of the Crank-Nicolson scheme
        x0_prior = inv_start_pt(y, x0_prior, P[1])

        #TODO come back to gradient initialisation
        skip = setup.skip_for_save
        _time = collect(Iterators.flatten(p.tt[1:skip:end-1] for p in P))

        θ_history = ParamHistory(setup)

        (new{ObsScheme,S,TX,TW,R}(Wnr, XXᵒ, XX, WWᵒ, WW, Pᵒ, P, fpt, ρ,
                                  check_if_recompute_ODEs(setup)
                                  AccptTracker(setup), θ_history,
                                  ActionTracker(setup), skip,
                                  setup.blocking == NoBlocking(), [], _time),
         ll, x0_prior, last(θ_history))
    end

    function Workspace(ws::Workspace{ObsScheme,S,TX,TW,R}, new_ρ::Float64
                       ) where {ObsScheme,S,TX,TW,R}
        new{ObsScheme,S,TX,TW,R}(ws.Wnr, ws.XXᵒ, ws.XX, ws.WWᵒ, ws.WW,
                                 ws.Pᵒ, ws.P, ws.fpt, new_ρ)
    end
end

act(action, ws::Workspace, i) = act(action, ws.action_tracker, i)


"""
    savePath!(Paths, XX, saveMe, skip)

If `saveMe` flag is true, then save the entire path spanning all segments in
`XX`. Only 1 in  every `skip` points is saved to reduce storage space.
"""
function save_path!(ws, wsXX, bXX) #TODO deprecate bXX
    XX = ws.no_blocking_used ? wsXX : bXX
    skip = ws.skip_for_save
    push!(ws.paths, collect(Iterators.flatten(XX[i].yy[1:skip:end-1]
                                               for i in 1:length(XX))))
end


struct ParamUpdtDefn{R,S,ST,T,U}
    updt_type::R
    updt_coord::S
    t_kernel::T
    priors::U
    recompute_ODEs::Bool

    function ParamUpdtDefn(updt_type::R, updt_coord::S, t_kernel::T, priors::U,
                           recompute_ODEs::Bool, ::ST)
        where {R<:ParamUpdateType,S,T,U,ST<:ODESolverType}
        new{R,S,ST,T,U}(updt_type, updt_coord, t_kernel, priors, recompute_ODEs)
    end
end

struct GibbsDefn{N}
    updates::NTuple{N,ParamUdptDefn}
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

getindex(g::GibbsDefn, i::Int) = g.updates[i]

length(g::GibbsDefn{N}) where N = N
