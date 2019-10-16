#=
    ----------------------------------------------------------------------------
    Implements the main functionality of the package: Markov chain Monte Carlo
    sampler, which allows for exploration of a product space between a path
    space (on which the diffusion of interest is defined) and a parameter space.
    The main function is `mcmc(...)` and it is supported by two main types of
    routines: `impute!(...)` and `update_param(...)`.
    ----------------------------------------------------------------------------
=#

import ForwardDiff: value # currently not needed, but will be


#===============================================================================
                                Main routine
===============================================================================#
"""
    mcmc(setup)

Gibbs sampler alternately imputing unobserved parts of the path and updating
unknown coordinates of the parameter vector. `setup` defines all variables of
the Markov Chain
"""
function mcmc(setup::MCMCSetup)
    adaptive_prop, num_mcmc_steps = setup.adaptive_prop, setup.num_mcmc_steps
    ws, ll, θ = Workspace(setup)
    gibbs = GibbsDefn(setup)
    init_adaptation!(adaptive_prop, ws)

    for i in 1:num_mcmc_steps
        verbose = act(Verbose(), ws, i)
        act(SavePath(), ws, i) && save_path!(ws)
        ws = next_set_of_blocks(ws)
        ll, acc = impute!(ws, ll, verbose, i)
        update!(ws.accpt_tracker, Imputation(), acc)
        act(Readjust(), ws, i) && readjust_pCN!(ws, i)

        if act(ParamUpdate(), ws, i)
            for j in 1:length(gibbs)
                ll, acc, θ = update_param!(gibbs[j], θ, ws, ll, verbose, i)
                update!(ws.accpt_tracker, ParamUpdate(), j, acc)
                update!(ws.θ_chain, θ)
                verbose && print("\n")
            end
            verbose && print("------------------------------------------------",
                             "------\n")
        end
        add_path!(adaptive_prop, ws.XX, i)
        #print_adaptation_info(adaptive_prop, accImpCounter, accUpdtCounter, i)
        adaptive_prop, ll = update!(adaptive_prop, ws, i, ll)
        adaptive_prop = still_adapting(adaptive_prop)
    end
    display_acceptance_rate(ws.blocking)
    ws
end

#===============================================================================
                            Imputation routines       (some also for param updt)
===============================================================================#

"""
    impute!(ws::Workspace{OS,NoBlocking}, ll, verbose=false, it=NaN,
            headstart=false) where OS

Imputation step of the MCMC scheme (without blocking).
...
# Arguments
- `ws`: workspace containing most containers and definitions of the chain
- `ll`: log-likelihood of the old (previously accepted) diffusion path
- `verbose`: whether to print update info while sampling
- `it`: iteration index of the MCMC algorithm
- `headstart`: flag for whether to 'ease into' fpt conditions
...
"""
function impute!(ws::Workspace{OS,NoBlocking}, ll, verbose=false, it=NaN,
                 headstart=false) where OS
    ρ = ws.ρ[1][1]
    # sample proposal starting point
    zᵒ, yᵒ = proposal_start_pt(ws, ws.P[1], ρ)

    # sample proposal path
    m = length(ws.WWᵒ)
    sample_segments!(1:m, ws, yᵒ, ρ, Val{headstart}())

    llᵒ = logpdf(ws.x0_prior, yᵒ)
    llᵒ += path_log_likhd(OS(), ws.XXᵒ, ws.P, 1:m, ws.fpt)
    llᵒ += lobslikelihood(ws.P[1], yᵒ)

    print_info(verbose, it, value(ll), value(llᵒ), "impute")

    if accept_sample(llᵒ-ll, verbose)
        swap!(ws.XX, ws.XXᵒ, ws.WW, ws.WWᵒ, 1:m)
        set!(ws.z, zᵒ)
        return llᵒ, true
    else
        return ll, false
    end
end

"""
    proposal_start_pt(ws::Workspace, P, ρ)

Sample a new proposal starting point using a random walk with pCN step
...
# Arguments
- `ws`: Workspace, contains previously accepted `z` and the definition of a prior
- `P`: diffusion law according to which auxiliary posterior is to be computed
- `ρ`: memory parameter of the preconditioned Crank Nicolson scheme
...
"""
function proposal_start_pt(ws::Workspace, P, ρ)
    x0_prior, z = ws.x0_prior, ws.z.val
    zᵒ = rand(x0_prior, z, ρ)
    yᵒ = start_pt(zᵒ, x0_prior, P)
    zᵒ, yᵒ
end


"""
    sample_segments!(iRange, ws::Workspace{OS}, y, ρ,
                     headstart::Val{false}=Val{false}()
                     ) where OS

Sample paths segments in index range `iRange` using preconditioned
Crank-Nicolson scheme
...
# Arguments
- `iRange`: range of indices of the segments that need to be sampled
- `ws`: Workspace, containing all relevant containers of the Markov chain
- `y`: starting point
- `ρ`: memory parameter of the preconditioned Crank-Nicolson scheme
- `headstart`: whether to ease into first-passage time sampling
...
"""
function sample_segments!(iRange, ws::Workspace{OS}, y, ρ,
                          headstart::Val{false}=Val{false}()) where OS
    for i in iRange
        y = sample_segment!(i, ws, y, ρ)
    end
end

function sample_segments!(iRange, ws::Workspace{OS}, y, ρ,
                          headstart::Val{true}=Val{false}()) where OS
    for i in iRange
        sample_segment!(i, ws, y, ρ)
        while !checkFpt(OS(), ws.XXᵒ[i], ws.fpt[i])
            sample_segment!(i, ws, y, ρ)
        end
        y = ws.XXᵒ[i].yy[end]
    end
end


"""
    sample_segment!(i, ws, y, ρ)

Sample the `i`th path segment using preconditioned Crank-Nicolson scheme
...
# Arguments
- `i`: index of the segment to be sampled
- `ws`: Workspace, containing all relevant containers of the Markov chain
- `y`: starting point
- `ρ`: memory parameter of the preconditioned Crank-Nicolson scheme
...
"""
function sample_segment!(i, ws, y, ρ)
    sample!(ws.WWᵒ[i], ws.Wnr)
    crank_nicolson!(ws.WWᵒ[i].yy, ws.WW[i].yy, ρ)
    solve!(Euler(), ws.XXᵒ[i], y, ws.WWᵒ[i], ws.P[i]) # always according to trgt law
    ws.XXᵒ[i].yy[end]
end


"""
    crank_nicolson!(yᵒ, y, ρ)

Preconditioned Crank-Nicolson update with memory parameter `ρ`, previous vector
`y` and new vector `yᵒ`
"""
crank_nicolson!(yᵒ, y, ρ) = (yᵒ .= √(1-ρ)*yᵒ + √(ρ)*y)


"""
    path_log_likhd(::OS, XX, P, iRange, fpt; skipFPT=false
                   ) where OS <: AbstractObsScheme

Compute likelihood for path `XX` to be observed under `P`. Only segments with
index numbers in `iRange` are considered. `fpt` contains relevant info about
checks regarding adherence to first passage time pattern. `skipFPT` if set to
`true` can skip the step of checking adherence to fpt pattern (used for
conjugate updates, or any updates that keep `XX` unchanged)
"""
function path_log_likhd(::OS, XX, P, iRange, fpt; skipFPT=false
                        ) where OS <: AbstractObsScheme
    ll = 0.0
    for i in iRange
        ll += llikelihood(LeftRule(), XX[i], P[i])
    end
    !skipFPT && (ll = check_full_path_fpt(OS(), XX, iRange, fpt) ? ll : -Inf)
    !skipFPT && (ll += check_domain_adherence(P, XX, iRange) ? 0.0 : -Inf)
    ll
end


"""
    print_info(verbose::Bool, it::Integer, ll, llᵒ, msg="update")

Print information to the console about current likelihood values

...
# Arguments
- `verbose`: flag for whether to print anything at all
- `it`: iteration of the Markov chain
- `ll`: likelihood of the previous, accepted sample
- `llᵒ`: likelihood of the proposal sample
- `msg`: message to start with
...
"""
function print_info(verbose::Bool, it::Integer, ll, llᵒ, msg="update")
    verbose && print(msg, ": ", it, " ll ", round(ll, digits=3), " ",
                     round(llᵒ, digits=3), " diff_ll: ", round(llᵒ-ll,digits=3))
end


"""
    accept_sample(log_threshold, verbose=false)

Make a random MCMC decision for whether to accept a sample or reject it.
"""
function accept_sample(log_threshold, verbose=false)
    if rand(Exponential(1.0)) > -log_threshold # Reject if NaN
        verbose && print("\t ✓\n")
        return true
    else
        verbose && print("\t .\n")
        return false
    end
end


"""
    swap!(A, Aᵒ, iRange)

Swap contents between containers A & Aᵒ in the index range iRange
"""
function swap!(A, Aᵒ, iRange)
    for i in iRange
        A[i], Aᵒ[i] = Aᵒ[i], A[i]
    end
end


"""
    swap!(A, Aᵒ, B, Bᵒ, iRange)

Swap contents between containers A & Aᵒ in the index range iRange, do the same
for containers B & Bᵒ
"""
function swap!(A, Aᵒ, B, Bᵒ, iRange)
    swap!(A, Aᵒ, iRange)
    swap!(B, Bᵒ, iRange)
end


"""
    impute!(ws::Workspace{OS,ChequeredBlocking}, ll, verbose=false, it=NaN,
            headstart=false) where OS

Imputation step of the MCMC scheme (without blocking).
...
# Arguments
- `ws`: workspace containing most containers and definitions of the chain
- `ll`: log-likelihood of the old (previously accepted) diffusion path
- `verbose`: whether to print update info while sampling
- `it`: iteration index of the MCMC algorithm
- `headstart`: flag for whether to 'ease into' fpt conditions
...
"""
function impute!(ws::Workspace{OS,<:ChequeredBlocking}, ll, verbose=false,
                 it=NaN, headstart=false) where OS
    P, XXᵒ, XX = ws.P, ws.XXᵒ, ws.XX

    recompute_accepted_law!(ws)
    # proposal starting point:
    z_prop, y_prop = proposal_start_pt(ws, P[1], ws.ρ[ws.blidx][1])

    ll_total = 0.0
    for (block_idx, block) in enumerate(ws.blocking.blocks[ws.blidx])
        block_flag = Val{block[1]}()
        # previously accepted starting point
        y = XX[block[1]].yy[1]
        # proposal starting point for the block (can be non-y only for the first block)
        yᵒ = choose_start_pt(block_flag, y, y_prop)

        # sample path in block
        sample_segments!(block, ws, yᵒ, ws.ρ[ws.blidx][block_idx])
        set_end_pt_manually!(block_idx, block, ws)

        # starting point, path and observations contribution
        llᵒ = start_pt_log_pdf(block_flag, ws.x0_prior, yᵒ)
        llᵒ += path_log_likhd(OS(), XXᵒ, P, block, ws.fpt)
        llᵒ += lobslikelihood(P[block[1]], yᵒ)

        llPrev = start_pt_log_pdf(block_flag, ws.x0_prior, y)
        llPrev += path_log_likhd(OS(), XX, P, block, ws.fpt; skipFPT=true)
        llPrev += lobslikelihood(P[block[1]], y)

        print_info(verbose, it, value(llPrev), value(llᵒ), "impute")
        if accept_sample(llᵒ-llPrev, verbose)
            swap!(XX, XXᵒ, block)
            register_accpt!(ws, block_idx, true)
            set_z!(block_flag, ws, z_prop)
            ll_total += llᵒ
        else
            register_accpt!(ws, block_idx, false)
            ll_total += llPrev
        end
    end
    # acceptance indicator does not matter for sampling with blocking
    return ll_total, true
end


"""
    recompute_accepted_law!(ws::Workspace)

Recompute the (H, Hν, c) triplet as well as the noise that corresponds to law
`ws.P` that is obtained after switching blocks
"""
function recompute_accepted_law!(ws::Workspace)
    solve_back_rec!(ws, ws.P)         # compute (H, Hν, c) for given blocks
    noise_from_path!(ws, ws.XX, ws.WW, ws.P) # find noise WW that generates XX under 𝔅.P

    # compute white noise generating starting point under 𝔅
    z = inv_start_pt(ws.XX[1].yy[1], ws.x0_prior, ws.P[1])
    set!(ws.z, z)
end


"""
    solve_back_rec!(::NoBlocking, ws::Workspace{OS,B,ST}, P) where {OS,B,ST}

Solve backward recursion to find H, Hν and c, which together define r̃(t,x)
and p̃(x, 𝓓) under the auxiliary law, when no blocking is done
"""
function solve_back_rec!(::NoBlocking, ws::Workspace{OS,B,ST}, P) where {OS,B,ST}
    m = length(P)
    gpupdate!(P[m]; solver=ST())
    for i in (m-1):-1:1
        gpupdate!(P[i], P[i+1].H[1], P[i+1].Hν[1], P[i+1].c[1]; solver=ST())
    end
end


"""
    solve_back_rec!(ws::Workspace{OS,B,ST}, P) where {OS,B,ST}

Solve backward recursion to find H, Hν and c, which together define r̃(t,x)
and p̃(x, 𝓓) under the auxiliary law, when blocking is done
"""
function solve_back_rec!(ws::Workspace{OS,B,ST}, P) where {OS,B,ST}
    for block in reverse(ws.blocking.blocks[ws.blidx])
        gpupdate!(P[block[end]]; solver=ST())
        for i in reverse(block[1:end-1])
            gpupdate!(P[i], P[i+1].H[1], P[i+1].Hν[1], P[i+1].c[1]; solver=ST())
        end
    end
end


"""
    noise_from_path!(ws::Workspace, XX, WW, P)

Compute driving Wiener noise `WW` from path `XX` drawn under law `P`
"""
function noise_from_path!(ws::Workspace, XX, WW, P)
    𝔅 = ws.blocking
    for block in 𝔅.blocks[ws.blidx]
        for i in block
            inv_solve!(Euler(), XX[i], WW[i], P[i])
        end
    end
end


choose_start_pt(::Val{1}, y, yᵒ) = copy(yᵒ)
choose_start_pt(::Any, y, yᵒ) = copy(y)


"""
    set_end_pt_manually!(block_idx, block, ws::Workspace)

Manually set the end-point of the proposal path under blocking so that it agrees
with the end-point of the previously accepted path. If it is the last block,
then do nothing
"""
function set_end_pt_manually!(block_idx, block, ws::Workspace)
    if block_idx < length(ws.blocking.blocks[ws.blidx])
        ws.XXᵒ[block[end]].yy[end] = ws.XX[block[end]].yy[end]
    end
end


"""
    start_pt_log_pdf(::Val{1}, yPr::StartingPtPrior, y)

Compute the log-likelihood contribution of the starting point for a given prior
under a blocking scheme (intended to be used with a first block only)
"""
start_pt_log_pdf(::Val{1}, x0_prior::StartingPtPrior, y) = logpdf(x0_prior, y)

"""
    start_pt_log_pdf(::Any, yPr::StartingPtPrior, y)

Default contribution to log-likelihood from the startin point under blocking
"""
start_pt_log_pdf(::Any, x0_prior::StartingPtPrior, y) = 0.0


set_z!(::Val{1}, ws::Workspace, zᵒ) = set!(ws.z, zᵒ)

set_z!(::Any, ::Workspace, ::Any) = nothing

#===============================================================================
                            Parameter update routines
===============================================================================#

"""
    update_param!(pu::ParamUpdtDefn{MetropolisHastingsUpdt,UpdtIdx}, θ,
                  ws::Workspace{OS,NoBlocking}, ll, verbose=false, it=NaN
                  ) where {UpdtIdx,OS}
Update parameters
...
# Arguments
- `pu`: determines transition kernel, priors and which parameters to update
- `θ`: current value of the parameter
- `ws`: workspace with all necessary containers needed by the Markov chain
- `ll`: likelihood of the old (previously accepted) parametrisation
- `verbose`: whether to print updates info while sampling
- `it`: iteration index of the MCMC algorithm
...
"""
function update_param!(pu::ParamUpdtDefn{MetropolisHastingsUpdt,UpdtIdx}, θ,
                       ws::Workspace{OS,NoBlocking}, ll, verbose=false,
                       it=NaN) where {UpdtIdx,OS}
    WW, Pᵒ, P, XXᵒ, XX, fpt = ws.WW, ws.Pᵒ, ws.P, ws.XXᵒ, ws.XX, ws.fpt
    m = length(WW)
    θᵒ = rand(pu.t_kernel, θ, UpdtIdx())               # sample new parameter
    update_laws!(Pᵒ, θᵒ)
    pu.recompute_ODEs && solve_back_rec!(NoBlocking(), ws, Pᵒ) # compute (H, Hν, c)

    # find white noise which for a given θᵒ gives a correct starting point
    y = XX[1].yy[1]
    zᵒ = inv_start_pt(y, ws.x0_prior, Pᵒ[1])

    find_path_from_wiener!(XXᵒ, y, WW, Pᵒ, 1:m)

    llᵒ = logpdf(ws.x0_prior, y)
    llᵒ += path_log_likhd(OS(), XXᵒ, Pᵒ, 1:m, fpt)
    llᵒ += lobslikelihood(Pᵒ[1], y)

    print_info(verbose, it, ll, llᵒ)

    llr = ( llᵒ - ll + prior_kernel_contrib(pu.t_kernel, pu.priors, θ, θᵒ))

    # Accept / reject
    if accept_sample(llr, verbose)
        swap!(XX, XXᵒ, P, Pᵒ, 1:m)
        set!(ws.z, zᵒ)
        return llᵒ, true, θᵒ
    else
        return ll, false, θ
    end
end


"""
    update_laws!(Ps, θᵒ)

Set new parameter `θᵒ` for the laws in vector `Ps`
"""
function update_laws!(Ps, θᵒ)
    m = length(Ps)
    for i in 1:m
        Ps[i] = GuidPropBridge(Ps[i], θᵒ)
    end
end


"""
    find_path_from_wiener!(XX, y, WW, P, iRange)

Find path `XX` (that starts from `y`) that is generated under law `P` from the
Wiener process `WW`. Only segments with indices in range `iRange` are considered
"""
function find_path_from_wiener!(XX, y, WW, P, iRange)
    for i in iRange
        solve!(Euler(), XX[i], y, WW[i], P[i])
        y = XX[i].yy[end]
    end
end


"""
    prior_kernel_contrib(t_kern, priors, θ, θᵒ)

Contribution to the log-likelihood ratio from transition kernel `t_kern` and
`priors`.
"""
function prior_kernel_contrib(t_kern, priors, θ, θᵒ)
    llr = logpdf(t_kern, θᵒ, θ) - logpdf(t_kern, θ, θᵒ)
    for prior in priors
        llr += logpdf(prior, θᵒ) - logpdf(prior, θ)
    end
    llr
end


#NOTE blocking and no-blocking param update should be joined into one function
"""
    update_param!(pu::ParamUpdtDefn{MetropolisHastingsUpdt,UpdtIdx}, θ,
                  ws::Workspace{OS,B,ST}, ll, verbose=false, it=NaN
                  ) where {UpdtIdx,OS,B,ST}
Update parameters
"""
function update_param!(pu::ParamUpdtDefn{MetropolisHastingsUpdt,UpdtIdx},
                       θ, ws::Workspace{OS,B}, ll, verbose=false, it=NaN
                       ) where {UpdtIdx,OS,B}
    WW, Pᵒ, P, XXᵒ, XX, fpt = ws.WW, ws.Pᵒ, ws.P, ws.XXᵒ, ws.XX, ws.fpt
    m = length(P)
    θᵒ = rand(pu.t_kernel, θ, UpdtIdx())               # sample new parameter
    update_laws!(Pᵒ, θᵒ)                   # update law `Pᵒ` accordingly
    solve_back_rec!(ws, Pᵒ)                 # compute (H, Hν, c)

    y = XX[1].yy[1]
    zᵒ = inv_start_pt(y, ws.x0_prior, Pᵒ[1])

    llᵒ = logpdf(ws.x0_prior, y)
    for (block_idx, block) in enumerate(ws.blocking.blocks[ws.blidx])
        y = XX[block[1]].yy[1]
        find_path_from_wiener!(XXᵒ, y, WW, Pᵒ, block)
        set_end_pt_manually!(block_idx, block, ws)

        # Compute log-likelihood ratio
        llᵒ += path_log_likhd(OS(), XXᵒ, Pᵒ, block, ws.fpt)
        llᵒ += lobslikelihood(Pᵒ[block[1]], y)
    end
    print_info(verbose, it, ll, llᵒ)

    llr = ( llᵒ - ll + prior_kernel_contrib(pu.t_kernel, pu.priors, θ, θᵒ))

    # Accept / reject
    if accept_sample(llr, verbose)
        swap!(XX, XXᵒ, P, Pᵒ, 1:m)
        set!(ws.z, zᵒ)
        return llᵒ, true, θᵒ
    else
        return ll, false, θ
    end
end


"""
    update_param!(pu::ParamUpdtDefn{ConjugateUpdt,UpdtIdx}, θ,
                  ws::Workspace{OS,NoBlocking,ST}, ll, verbose=false, it=NaN
                  ) where {UpdtIdx,OS,ST}
Update parameters using conjugate draws
"""
function update_param!(pu::ParamUpdtDefn{ConjugateUpdt,UpdtIdx}, θ,
                       ws::Workspace{OS,NoBlocking}, ll, verbose=false, it=NaN
                       ) where {UpdtIdx,OS}
    WW, Pᵒ, P, XXᵒ, XX, fpt = ws.WW, ws.Pᵒ, ws.P, ws.XXᵒ, ws.XX, ws.fpt
    m = length(WW)
    θᵒ = conjugate_draw(θ, XX, P[1].Target, pu.priors[1], UpdtIdx())   # sample new parameter


    update_laws!(P, θᵒ)
    pu.recompute_ODEs && solve_back_rec!(NoBlocking(), ws, P) # compute (H, Hν, c)

    for i in 1:m    # compute wiener path WW that generates XX
        inv_solve!(Euler(), XX[i], WW[i], P[i])
    end
    # compute white noise that generates starting point
    y = XX[1].yy[1]
    z = inv_start_pt(y, ws.x0_prior, P[1])

    llᵒ = logpdf(ws.x0_prior, y)
    llᵒ += path_log_likhd(OS(), XX, P, 1:m, fpt; skipFPT=true)
    llᵒ += lobslikelihood(P[1], y)
    print_info(verbose, it, value(ll), value(llᵒ))
    set!(ws.z, z)
    return llᵒ, true, θᵒ
end


#NOTE blocking and no-blocking param conjugate update should be joined into one function
"""
    update_param!(pu::ParamUpdtDefn{ConjugateUpdt,UpdtIdx}, θ,
                  ws::Workspace{OS,B}, ll, verbose=false, it=NaN
                  ) where {UpdtIdx,OS,B}
Update parameters
see the definition of  update_param!(…, ::MetropolisHastingsUpdt, …) for the
explanation of the arguments.
"""
function update_param!(pu::ParamUpdtDefn{ConjugateUpdt,UpdtIdx}, θ,
                       ws::Workspace{OS,B}, ll, verbose=false, it=NaN
                       ) where {UpdtIdx,OS,B}
    WW, Pᵒ, P, XXᵒ, XX, fpt = ws.WW, ws.Pᵒ, ws.P, ws.XXᵒ, ws.XX, ws.fpt
    m = length(WW)
    θᵒ = conjugate_draw(θ, XX, P[1].Target, pu.priors[1], UpdtIdx())   # sample new parameter

    update_laws!(P, θᵒ)
    pu.recompute_ODEs && solve_back_rec!(ws, P)
    for i in 1:m    # compute wiener path WW that generates XX
        inv_solve!(Euler(), XX[i], WW[i], P[i])
    end
    # compute white noise that generates starting point
    y = XX[1].yy[1]
    z = inv_start_pt(y, ws.x0_prior, P[1])

    llᵒ = logpdf(ws.x0_prior, y)
    for block in ws.blocking.blocks[ws.blidx]
        llᵒ += path_log_likhd(OS(), XX, P, block, ws.fpt; skipFPT=true)
        llᵒ += lobslikelihood(P[block[1]], XX[block[1]].yy[1])
    end
    print_info(verbose, it, value(ll), value(llᵒ))
    set!(ws.z, z)
    return llᵒ, true, θᵒ
end
