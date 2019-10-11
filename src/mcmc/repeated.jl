# using DelimitedFiles
# using Makie
# data = readdlm("../LinneasData190920.csv", ';')
#
# data[isnan.(data)] .= circshift(data, (-1,0))[isnan.(data)]
# data[isnan.(data)] .= circshift(data, (1,0))[isnan.(data)]
# data[isnan.(data)] .= circshift(data, (-2,0))[isnan.(data)]
# data[isnan.(data)] .= circshift(data, (2,0))[isnan.(data)]
# data[isnan.(data)] .= circshift(data, (3,0))[isnan.(data)]
# any(isnan.(data))
#
# #data = replace(data, NaN=>missing)
# #μ = mapslices(mean∘skipmissing, data, dims=1)
# #sigma = mapslices(std∘skipmissing, data, dims=1)
# #surface(0..1, 0..5, data)

# NOTE there is some loss of generality with the way recompute_ODEs is defined
# recompute_ODEs is the same across all samples, in previous implementation
# it could be different, to remedy this, adjust the definition of GibbsDefn in
# workspace.jl
"""
    mcmc(setups::Vector{<:MCMCSetup})

Gibbs sampler alternately imputing unobserved parts of the paths and updating
unknown coordinates of the parameter vector. Version suitable for multiple
trajectory samples. `setups` defines all variables required for the
initialisation of the Markov Chain
"""
function mcmc(setups::Vector{<:MCMCSetup})
    num_mcmc_steps, K = setups[1].num_mcmc_steps, length(setups)
    tu = Workspace(setups[1])
    ws, ll, θ = [tu.workspace], [tu.ll], [tu.θ]
    for k in 2:K
        tu = Workspace(setups[k])
        push!(ws, tu.workspace); push!(ll, tu.ll); push!(θ, tu.θ)
    end
    gibbs = GibbsDefn(setups[1])

    for i in 1:num_mcmc_steps
        verbose = act(Verbose(), ws[1], i)
        act(SavePath(), ws[1], i) && for k in 1:K save_path!(ws[k]) end
        for k in 1:K next_set_of_blocks(ws[k]) end
        for k in 1:K
            ll[k], acc = impute!(ws[k], ll[k], verbose, i)
            update!(ws[k].accpt_tracker, Imputation(), acc)
        end

        if act(ParamUpdate(), ws[1], i)
            for j in 1:length(gibbs)
                ll, acc, θ = update_param!(gibbs[j], θ, ws, ll, verbose, i)

                for k in 1:K
                    update!(ws[k].accpt_tracker, ParamUpdate(), j, acc)
                    update!(ws[k].θ_chain, θ[k])
                end
                verbose && print("\n")
            end
            P˟ = clone(ws[1].P[1].Target, θ[1])
            verbose && println(prod("$v=$x " for (v, x) in zip(param_names(P˟), orig_params(P˟))))
            verbose && print("------------------------------------------------",
                             "------\n")
        end
    end
    ws
end

# NOTE for the Mixed Effect models there will need to be some indicator matrix
# passed arround to indicate which indices are common and how they should be
# sampled. For the repeated.jl all θs that are passed around are the same, so
# for now I discard all the remaining θs [⋆]
function conjugate_draw(θ, XX::Vector{<:Vector}, PT, prior, updtIdx)
    μ = mustart(updtIdx)
    𝓦 = μ*μ'
    ϑ = SVector(thetaex(updtIdx, θ[1])) #NOTE [⋆] hence θ[1]
    for k in 1:length(XX)
        μ, 𝓦 = _conjugate_draw(ϑ, μ, 𝓦, XX[k], PT, updtIdx)
    end
    Σ = inv(𝓦 + inv(Matrix(prior.Σ)))
    Σ = (Σ + Σ')/2 # eliminates numerical inconsistencies
    μ_post = Σ * (μ + Vector(prior.Σ\prior.μ))
    ϑ = rand(Gaussian(μ_post, Σ))
    # expand back to one parameter per path
    [move_to_proper_place(ϑ, θ_k, updtIdx) for θ_k in θ]
end

# NOTE `ws` has UpdtIdx defined only as a type, can change it later to be a
# member so that passing ws[k].updtIdx instead of UpdtIdx() is possible
# NOTE same applies to OS (observation scheme)
#
# no blocking
function update_param!(pu::ParamUpdtDefn{ConjugateUpdt,UpdtIdx}, θ,
                       ws::Vector{<:Workspace{OS,NoBlocking}}, ll,
                       verbose=false, it=NaN) where {UpdtIdx,OS}
    K = length(ws)
    # warn if targets are different?
    θᵒ = conjugate_draw(θ, [w.XX for w in ws], ws[1].P[1].Target, pu.priors[1], UpdtIdx())   # sample new parameter

    for k in 1:K
        WW, Pᵒ, P, XXᵒ, XX = ws[k].WW, ws[k].Pᵒ, ws[k].P, ws[k].XXᵒ, ws[k].XX
        m = length(WW)
        update_laws!(P, θᵒ[k]) # hardcoded: NO Blocking
        pu.recompute_ODEs && solve_back_rec!(NoBlocking(), ws[k], P) # compute (H, Hν, c)

        for i in 1:m    # compute wiener path WW that generates XX
            inv_solve!(Euler(), XX[i], WW[i], P[i])
        end
        # compute white noise that generates starting point
        y = XX[1].yy[1]
        z = inv_start_pt(y, ws[k].x0_prior, P[1])

        llᵒ = logpdf(ws[k].x0_prior, y)
        llᵒ += path_log_likhd(OS(), XX, P, 1:m, ws[k].fpt; skipFPT=true)
        llᵒ += lobslikelihood(P[1], y)
        print_info(verbose, it, value(ll[k]), value(llᵒ))
        verbose && k < K && print("\n")
        ll[k] = llᵒ
        set!(ws[k].z, z)
    end
    return ll, true, θᵒ
end

function update_param!(pu::ParamUpdtDefn{MetropolisHastingsUpdt,UpdtIdx}, θ,
                      ws::Vector{<:Workspace{OS,NoBlocking}}, ll, verbose=false,
                      it=NaN) where {UpdtIdx,OS}
    K = length(ws)
    #NOTE for now, let's sample parameter corresponding to a single path, this
    # will need to be changed for Mixed effect models
    θᵒ = rand(pu.t_kernel, θ[1], UpdtIdx())               # sample new parameter
    llᵒ = copy(ll)
    llr = prior_kernel_contrib(pu.t_kernel, pu.priors, θ[1], θᵒ)
    zᵒ = [copy(w.z.val) for w in ws]
    for k in 1:K
        WW, Pᵒ, P, XXᵒ, XX = ws[k].WW, ws[k].Pᵒ, ws[k].P, ws[k].XXᵒ, ws[k].XX
        m = length(WW)
        update_laws!(Pᵒ, θᵒ)
        pu.recompute_ODEs && solve_back_rec!(NoBlocking(), ws[k], Pᵒ) # compute (H, Hν, c)

        # find white noise which for a given θᵒ gives a correct starting point
        y = XX[1].yy[1]
        zᵒ[k] = inv_start_pt(y, ws[k].x0_prior, Pᵒ[1])

        find_path_from_wiener!(XXᵒ, y, WW, Pᵒ, 1:m)

        llᵒ[k] = logpdf(ws[k].x0_prior, y)
        llᵒ[k] += path_log_likhd(OS(), XXᵒ, Pᵒ, 1:m, ws[k].fpt)
        llᵒ[k] += lobslikelihood(Pᵒ[1], y)

        print_info(verbose, it, ll[k], llᵒ[k])
        verbose && k < K && print("\n")
        llr += llᵒ[k] - ll[k]
    end

    # Accept / reject
    if accept_sample(llr, verbose)
        for k in 1:K
            m = length(ws[k].WW)
            swap!(ws[k].XX, ws[k].XXᵒ, ws[k].P, ws[k].Pᵒ, 1:m)
            set!(ws[k].z, zᵒ[k])
        end
        # expand back to a vector of parameters
        return llᵒ, true, [copy(θᵒ) for _ in θ]
    else
        return ll, false, θ
    end
end
