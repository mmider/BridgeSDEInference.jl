# NOTE in mcmc.jl

using ForwardDiff
using ForwardDiff: value, Dual

CTAG = ForwardDiff.Tag{Val{:custom_tag}, Float64}

#TODO need to pause langevin updates for a moment
"""
    findProposalLaw(xx, tt, P˟, P̃, Ls, Σs; dt=1/5000, timeChange=true,
                    solver::ST=Ralston3())

Initialise the object with proposal law and all the necessary containers needed
for the simulation of the guided proposals
"""
function findProposalLaw(::LangevinUpdt, ::Type{K}, xx, tt, P˟, P̃, Ls, Σs, τ; dt=1/5000,
                         solver::ST=Ralston3(),
                         changePt::ODEChangePt=NoChangePt()) where {K,ST}
    m = length(xx) - 1
    P = Array{ContinuousTimeProcess,1}(undef,m)
    params(Pˣ)
#    P˟_D = clone(Pˣ, )#TODO here is stop point
    for i in m:-1:1
        numPts = Int64(ceil((tt[i+1]-tt[i])/dt))+1
        t = τ(tt[i], tt[i+1]).( range(tt[i], stop=tt[i+1], length=numPts) )
        xx_D = Dual{CT}.(xx[i+1])
        L_D = Dual{CT}.(Ls[i])
        Σ_D = Dual{CT}.(Σs[i])

        P[i] = ( (i==m) ? GuidPropBridge(K, t, P˟, P̃[i], L_D, xx_D, Σ_D;
                                         changePt=changePt, solver=ST()) :
                          GuidPropBridge(K, t, P˟, P̃[i], L_D, xx_D, Σ_D,
                                         P[i+1].H[1], P[i+1].Hν[1], P[i+1].c[1];
                                         changePt=changePt, solver=ST()) )
    end
    P
end


function prepareLangevin(𝓦𝓢::Workspace{ObsScheme}, θ, ::UpdtIdx, y, m, yPr,
                         priors, ::ST, uidx) where {ObsScheme,UpdtIdx,ST}
    idxToUpdt = idx(UpdtIdx())
    function _ll(ϑ)
        XX, WW, P, fpt = 𝓦𝓢.XX, 𝓦𝓢.WW, 𝓦𝓢.P, 𝓦𝓢.fpt
        updateLaws!(P, ϑ)
        solveBackRec!(NoBlocking(), P, ST()) # changes nothing, but needed for ∇
        findPathFromWiener!(XX, y, WW, P, 1:m)

        ll = logpdf(yPr, y)
        ll += pathLogLikhd(ObsScheme(), XX, P, 1:m, fpt)
        ll += lobslikelihood(P[1], y)
        for prior in priors
            ll += logpdf(prior, θ)
        end
        ll
        ϑ[1]
    end
    ϑ = [θ[i] for i in idxToUpdt]
    chunkSize = 1
    result = 𝓦𝓢.result[uidx]
    cfg = ForwardDiff.GradientConfig(_ll, ϑ, ForwardDiff.Chunk{chunkSize}())
    ForwardDiff.gradient!(result, _ll, ϑ, cfg)
    DiffResults.value(result), DiffResults.gradient(result)
end


function postProcessLangevin(𝓦𝓢::Workspace{ObsScheme}, θᵒ, ::UpdtIdx, y, m, yPr,
                         priors, ::ST, uidx) where {ObsScheme,UpdtIdx,ST}
    idxToUpdt = idx(UpdtIdx())
    function _ll(ϑ)
        XXᵒ, WW, Pᵒ, fpt = 𝓦𝓢.XXᵒ, 𝓦𝓢.WW, 𝓦𝓢.Pᵒ, 𝓦𝓢.fpt
        for (i, ui) in enumerate(idxToUpdt)
            θᵒ[ui] = ϑ[i]
        end
        updateLaws!(Pᵒ, θᵒ)
        solveBackRec!(NoBlocking(), Pᵒ, ST()) # changes nothing, but needed for ∇
        findPathFromWiener!(XX, y, WW, P, 1:m)

        yPrᵒ = invStartPt(y, yPr, Pᵒ[1])

        ll = logpdf(yPrᵒ, y)
        ll += pathLogLikhd(ObsScheme(), XXᵒ, Pᵒ, 1:m, fpt)
        ll += lobslikelihood(Pᵒ[1], y)
        for prior in priors
            ll += logpdf(prior, θᵒ)
        end
        ll
        ϑ[2]
    end
    ϑ = [θᵒ[i] for i in idxToUpdt]
    chunkSize = 1
    result = 𝓦𝓢.resultᵒ[uidx]
    cfg = ForwardDiff.GradientConfig(_ll, ϑ, ForwardDiff.Chunk{chunkSize}())
    ForwardDiff.gradient!(result, _ll, ϑ, cfg)

    yPrᵒ = invStartPt(y, yPr, 𝓦𝓢.Pᵒ[1])
    DiffResults.value(result), DiffResults.gradient(result), yPrᵒ
end


function updateParam!(::LangevinUpdt, 𝔅::NoBlocking, tKern, θ,
                      ::UpdtIdx, yPr, 𝓦𝓢::Workspace{ObsScheme}, ll, priors,
                      recomputeODEs; solver::ST=Ralston3(), verbose=false,
                      it=NaN, uidx=NaN) where {ObsScheme <: AbstractObsScheme, ST, UpdtIdx}
    WW, Pᵒ, P, XXᵒ, XX, fpt = 𝓦𝓢.WW, 𝓦𝓢.Pᵒ, 𝓦𝓢.P, 𝓦𝓢.XXᵒ, 𝓦𝓢.XX, 𝓦𝓢.fpt
    m = length(WW)
    y = XX[1].yy[1]
    ll, ∇ll = prepareLangevin(𝓦𝓢, θ, UpdtIdx(), y, m, yPr, priors, ST(), uidx) # TODO pre-allocate ∇ll
    θᵒ = rand(tKern, θ, ∇ll, UpdtIdx())               # sample new parameter
    llᵒ, ∇llᵒ, yPrᵒ = postProcessLangevin(𝓦𝓢, θᵒ, UpdtIdx(), y, m, yPr, priors, ST(), uidx)

    printInfo(verbose, it, ll, llᵒ)

    llr = ( llᵒ - ll + logpdf(tKern, θᵒ, θ, ∇llᵒ, UpdtIdx()) - logpdf(tKern, θ, θᵒ, ∇ll, UpdtIdx()))

    # Accept / reject
    if acceptSample(llr, verbose)
        swap!(XX, XXᵒ, P, Pᵒ, 1:m)
        swap!(𝓦𝓢.resultᵒ, 𝓦𝓢.result, 1:m)
        return llᵒ, true, θᵒ, yPrᵒ
    else
        return ll, false, θ, yPr
    end
end
