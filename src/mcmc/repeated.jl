function mcmc(setup_mcmc::MCMCSetup, schedule::MCMCSchedule, setups::Vector{T}) where T <: ModelSetup
    workspaces = [create_workspace(setup) for setup in setups]
    wss, lls = [ws[1] for ws in workspaces], [ws[2] for ws in workspaces]
    θ = workspaces[1][3]
    ws_mcmc = create_workspace(setup_mcmc, schedule, θ)
    # adpt = adaptation_object(setup, ws) # adaptive auxiliary law currently not supported

    aux = nothing
    for step in schedule
        step.save && [save_imputed!(ws) for ws in wss]
        for i in step.idx
            if step.param_updt || typeof(ws_mcmc.updates[i]) <: Imputation
                wss = [next(ws, ws_mcmc.updates[i]) for ws in wss]
                lls, acc, θ = update!(ws_mcmc.updates[i], wss, θ, lls, step, aux)
                aux = aux_params(ws_mcmc.updates[i], aux)
                #TODO: make it nicer, currently we need a little dance here
                dance = typeof(ws_mcmc.updates[i]) <: Imputation
                dance && begin
                    update!(ws_mcmc, acc[1], θ, i)
                    for k in 2:length(setups)
                        register_accpt!(ws_mcmc.updates[i], acc[k])
                    end
                end
                !dance && update!(ws_mcmc, acc, θ, i)
                step.verbose && print("\n")
            end
        end
        step.verbose && print("-----------------------------------------------",
                              "------\n")
        step.readjust && readjust!(ws_mcmc, step.iter)
        step.fuse && fuse!(ws_mcmc, schedule)
        # ll = adaptation!(ws, adpt, step.iter, ll) adaptation currently not supported
    end
    [display_summary(ws, ws_mcmc) for ws in wss]
    wss, ws_mcmc
end

function update!(updt::Imputation{NoBlocking}, wss::Vector{<:Workspace{OS}}, θ,
                 lls, step, ::Any, headstart=false) where OS
    ρ, K = updt.ρs[1], length(wss)
    accpts = repeat([false], K)

    for k in 1:K
        ws = wss[k]
        # sample proposal starting point
        zᵒ, yᵒ = proposal_start_pt(ws, ws.P[1], ρ)

        # sample proposal path
        m = length(ws.WWᵒ)
        success = sample_segments!(1:m, ws, yᵒ, ρ, Val{headstart}())
        llᵒ = success ? (logpdf(ws.x0_prior, yᵒ) +
                         path_log_likhd(OS(), ws.XXᵒ, ws.P, 1:m, ws.fpt) +
                         lobslikelihood(ws.P[1], yᵒ)) : -Inf

        print_info(step, value(lls[k]), value(llᵒ), "impute")

        if accept_sample(llᵒ-lls[k], step.verbose)
            swap!(ws.XX, ws.XXᵒ, ws.WW, ws.WWᵒ, 1:m)
            set!(ws.z, zᵒ)
            lls[k] = llᵒ
            accpts[k] = true
        end
    end
    lls, accpts, θ
end

function update!(pu::ParamUpdate{MetropolisHastingsUpdt},
                 wss::Vector{<:Workspace{OS}}, θ, lls, step,
                 blocking::NoBlocking) where OS
    K = length(wss)

    #NOTE for now, let's sample parameter corresponding to a single path, this
    # will need to be changed for Mixed effect models
    θᵒ = rand(pu.t_kernel, θ, pu.updt_coord)               # sample new parameter
    (logpdf(pu.priors, θᵒ) === -Inf) && (return lls, false, θ)

    llᵒs = copy(lls)
    zᵒs = [copy(ws.z.val) for ws in wss]
    for k in 1:K
        ws = wss[k]
        WW, Pᵒ, P, XXᵒ, XX, fpt = ws.WW, ws.Pᵒ, ws.P, ws.XXᵒ, ws.XX, ws.fpt
        m = length(WW)
        update_laws!(Pᵒ, θᵒ)
        pu.aux.recompute_ODEs && solve_back_rec!(blocking, pu.aux.solver, Pᵒ) # compute (H, Hν, c)

        # find white noise which for a given θᵒ gives a correct starting point
        y = XX[1].yy[1]
        zᵒs[k] = inv_start_pt(y, ws.x0_prior, Pᵒ[1])

        success = find_path_from_wiener!(XXᵒ, y, WW, Pᵒ, 1:m)

        llᵒs[k] = success ? (logpdf(ws.x0_prior, y) +
                             path_log_likhd(OS(), XXᵒ, Pᵒ, 1:m, fpt) +
                             lobslikelihood(Pᵒ[1], y)) : -Inf
    end
    ll, llᵒ = sum(lls), sum(llᵒs)
    print_info(step, value(ll), value(llᵒ))

    llr = ( llᵒ - ll + prior_kernel_contrib(pu, θ, θᵒ))

    # Accept / reject
    if accept_sample(llr, step.verbose)
        for k in 1:K
            ws = wss[k]
            WW, Pᵒ, P, XXᵒ, XX, fpt = ws.WW, ws.Pᵒ, ws.P, ws.XXᵒ, ws.XX, ws.fpt
            m = length(WW)
            swap!(XX, XXᵒ, P, Pᵒ, 1:m)
            set!(ws.z, zᵒs[k])
        end
        return llᵒs, true, θᵒ
    else
        return lls, false, θ
    end
end


function update!(pu::ParamUpdate{ConjugateUpdt}, wss::Vector{<:Workspace{OS}},
                 θ, lls, step, blocking::NoBlocking) where OS
    K = length(wss)

    θᵒ = conjugate_draw(θ, [ws.XX for ws in wss], wss[1].P[1].Target, pu.priors[1], pu.updt_coord)

    total_ll_old = sum(lls)
    for k in 1:K
        ws = wss[k]
        WW, P, XX, fpt = ws.WW, ws.P, ws.XX, ws.fpt
        m = length(WW)

        update_laws!(P, θᵒ)
        pu.aux.recompute_ODEs && solve_back_rec!(blocking, pu.aux.solver, P) # compute (H, Hν, c)

        for i in 1:m    # compute wiener path WW that generates XX
            inv_solve!(EulerMaruyamaBounded(), XX[i], WW[i], P[i])
        end
        # compute white noise that generates starting point
        y = XX[1].yy[1]
        z = inv_start_pt(y, ws.x0_prior, P[1])

        llᵒ = logpdf(ws.x0_prior, y)
        llᵒ += path_log_likhd(OS(), XX, P, 1:m, fpt; skipFPT=true)
        llᵒ += lobslikelihood(P[1], y)
        lls[k] = llᵒ
        set!(ws.z, z)
    end
    print_info(step, sum(total_ll_old), sum(lls))
    return lls, true, θᵒ
end

function conjugate_draw(θ, XX::Vector{<:Vector}, PT, prior, updtIdx)
    μ = mustart(updtIdx)
    𝓦 = μ*μ'
    ϑ = SVector(thetaex(updtIdx, θ))
    for X in XX
        μ, 𝓦 = _conjugate_draw(ϑ, μ, 𝓦, X, PT, updtIdx)
    end
    Σ = inv(𝓦 + inv(Matrix(prior.Σ)))
    Σ = (Σ + Σ')/2 # eliminates numerical inconsistencies
    μ_post = Σ * (μ + Vector(prior.Σ\prior.μ))
    ϑ = rand(Gaussian(μ_post, Σ))
    move_to_proper_place(ϑ, θ, updtIdx)     # align so that dimensions agree
end
