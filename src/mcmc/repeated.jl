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


function mcmc(setups::Vector{MCMCSetup})
    num_mcmc_steps, K = setups[1].num_mcmc_steps, length(setups)
    tu = Workspace(setups[k])
    ws, ll, θ = tu.workspace, tu.ll, tu.θ
    for k in 2:K
        tu = Workspace(setups[k])
        push!(ws, tu.workspace); push!(ll, tu.ll); push!(θ, tu.θ)
    end

    #=
    P = [findProposalLaw(𝕂, obs[k], obsTimes[k], P˟, P̃[k], Ls[k], Σs[k], τ; dt=dt, solver=solver,
                     changePt=CP(getChangePt(blockingParams[3])) ) for k in 1:K]

    updtLen = length(updtCoord)
    tu = initialise(obsScheme, P[1], length(obs[1]) - 1, yPr[1], w, fpt[1])
    Wnr = [tu[1]]; WWᵒ = [tu[2]]; WW = [tu[3]];
    XXᵒ= [tu[4]]; XX = [tu[5]]; Pᵒ = [tu[6]];
    ll = [tu[7]]
    yPr[1] = tu[8]
    for k in 2:K
        tu = initialise(obsScheme, P[k], length(obs[k]) - 1, yPr[k], w, fpt[k])
        push!(Wnr, tu[1]); push!(WWᵒ, tu[2]); push!(WW, tu[3]);
        push!(XXᵒ, tu[4]); push!(XX, tu[5]); push!(Pᵒ, tu[6]);
        push!(ll, tu[7]);
        yPr[k] = tu[end]
    end

    Paths = []
    accImpCounter = 0
    accUpdtCounter = [0 for i in 1:updtLen]
    θ = params(P˟)
    θchain = Vector{typeof(θ)}(undef, (numSteps-warmUp)*updtLen+1)
    θchain[1] = copy(θ)

    recomputeODEs = [any([e in dependsOnParams(P[1][1].Pt) for e
                         in idx(uc)]) for uc in updtCoord]

    updtStepCounter = 1
    𝔅 = [setBlocking(blocking, blockingParams, P[k], WW[k], XX[k]) for k in 1:K]
    #display(𝔅)
    acc = zeros(Bool, K)
    =#
    for i in 1:num_mcmc_steps
        verbose = act(Verbose(), ws[1], i)#(i % verbIter == 0)
    #    i > warmUp && savePath!(Paths, blocking == NoBlocking() ? XX : 𝔅.XX,
#                                (i % saveIter == 0), skipForSave)
        act(SavePath(), ws[1], i) && for k in 1:K save_path!(ws[k]) end
        for k in 1:K next_set_of_blocks(ws[k]) end

        for k in 1:K

            tu = impute!(obsScheme, 𝔅[k], Wnr[k], yPr[k], WWᵒ[k], WW[k], XXᵒ[k], XX[k],
                                  P[k], ll[k], fpt[k], ρ=ρ, verbose=verbose, it=i,
                                  solver=solver)
            ll[k] = tu[1]; acc[k] = tu[2]; 𝔅[k] = tu[3]; yPr[k] = tu[4]
        end
        accImpCounter += sum(acc)
        if paramUpdt && i > warmUp
            for j in 1:updtLen
                ll, accp, θ, yPr = updateParam!(obsScheme, updtType[j], 𝔅, tKernel, θ,
                                     updtCoord[j], yPr, WW, Pᵒ, P, XXᵒ, XX, ll,
                                     priors[j], fpt, recomputeODEs[j];
                                     solver=solver, verbose=verbose, it=i)

                P˟ = clone(P˟, θ)
                accUpdtCounter[j] += 1*accp
                updtStepCounter += 1
                θchain[updtStepCounter] = copy(θ)
                verbose && print("\n")
            end
            verbose && println(prod("$v=$x " for (v, x) in zip(param_names(P˟), orig_params(P˟))))
            verbose && print("------------------------------------------------",
                             "------\n")
        end
    end
#    displayAcceptanceRate(𝔅)
#    Time = [collect(Iterators.flatten(p.tt[1:skipForSave:end-1] for p in P)) for P in PP]
    θchain, accImpCounter/numSteps, accUpdtCounter./numSteps#, Paths, Time
end

function conjugateDraw(θ, XX::Vector{<:Vector}, PT, prior, updtIdx)
    μ = mustart(updtIdx)
    𝓦 = μ*μ'
    ϑ = SVector(thetaex(updtIdx, θ))
    for k in 1:length(XX)
        μ, 𝓦 = _conjugateDraw(ϑ, μ, 𝓦, XX[k], PT, updtIdx)
    end
    Σ = inv(𝓦 + inv(Matrix(prior.Σ)))
    Σ = (Σ + Σ')/2 # eliminates numerical inconsistencies
    μ_post = Σ * (μ + Vector(prior.Σ\prior.μ))
    rand(Gaussian(μ_post, Σ))
end


# no blocking
function updateParam!(obsScheme::AbstractObsScheme, ::ConjugateUpdt, 𝔅::Vector{<:NoBlocking},
                      tKern, θ, updtIdx, yPr, WW, Pᵒ, P, XXᵒ, XX, ll::Vector, priors,
                      fpt, recomputeODEs; solver=Ralston3(), verbose=false,
                      it=NaN)
    K = length(P)
    # warn if targets are different?
    ϑ = conjugateDraw(θ, XX, P[1][1].Target, priors[1], updtIdx)   # sample new parameter
    θᵒ = moveToProperPlace(ϑ, θ, updtIdx)     # align so that dimensions agree
    for k in 1:K
        m = length(P[k])
        updateLaws!(P[k], θᵒ) # hardcoded: NO Blocking
        recomputeODEs && solveBackRec!(𝔅[k], P[k], solver) # compute (H, Hν, c)

        for i in 1:m    # compute wiener path WW that generates XX
            invSolve!(Euler(), XX[k][i], WW[k][i], P[k][i])
        end
        # compute white noise that generates starting point
        y = XX[k][1].yy[1]
        yPr[k] = invStartPt(y, yPr[k], P[k][1])

        ll[k] = logpdf(yPr[k], y)
        ll[k] += pathLogLikhd(obsScheme, XX[k], P[k], 1:m, fpt[k]; skipFPT=true)
        ll[k] += lobslikelihood(P[k][1], y)
    end

    #printInfo(verbose, it, value(ll), value(llᵒ))
    return ll, true, θᵒ, yPr
end

function updateParam!(obsScheme::AbstractObsScheme, ::MetropolisHastingsUpdt, 𝔅::Vector{<:NoBlocking},
                      tKern, θ, updtIdx, yPr, WW, Pᵒ, P, XXᵒ, XX, ll::Vector, priors,
                      fpt, recomputeODEs; solver=Ralston3(), verbose=false,
                      it=NaN)
    K = length(P)
    θᵒ = rand(tKern, θ, updtIdx)               # sample new parameter
    llᵒ = copy(ll)
    yPrᵒ = copy(yPr)
    llr = priorKernelContrib(tKern, priors, θ, θᵒ)
    for k in 1:K
        m = length(WW[k])
        updateLaws!(Pᵒ[k], θᵒ)
        recomputeODEs && solveBackRec!(𝔅[k], Pᵒ[k], solver) # compute (H, Hν, c)

    # find white noise which for a given θᵒ gives a correct starting point
        y = XX[k][1].yy[1]
        yPrᵒ[k] = invStartPt(y, yPr[k], Pᵒ[k][1])

        findPathFromWiener!(XXᵒ[k], y, WW[k], Pᵒ[k], 1:m)

        llᵒ[k] = logpdf(yPrᵒ[k], y)
        llᵒ[k] += pathLogLikhd(obsScheme, XXᵒ[k], Pᵒ[k], 1:m, fpt[k])
        llᵒ[k] += lobslikelihood(Pᵒ[k][1], y)

        printInfo(verbose, it, ll[k], llᵒ[k])
        llr += llᵒ[k] - ll[k]
    end

    # Accept / reject
    if acceptSample(llr, verbose)
        for k in 1:K
            m = length(WW[k])
            swap!(XX[k], XXᵒ[k], P[k], Pᵒ[k], 1:m)
        end
        #ll .= llᵒ
        return llᵒ, true, θᵒ, yPrᵒ
    else
        return ll, false, θ, yPr
    end
end
