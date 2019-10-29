function pseudo_conjugate_draw(θ, XX, PT, prior, updtIdx)
    μ = mustart(updtIdx)
    𝓦 = μ*μ'
    ϑ = SVector(thetaex(updtIdx, θ))
    μ, 𝓦 = _conjugate_draw(ϑ, μ, 𝓦, XX, PT, updtIdx)

    Σ = inv(𝓦 + inv(Matrix(prior.Σ)))
    Σ = (Σ + Σ')/2 # eliminates numerical inconsistencies
    μ_post = Σ * (μ + Vector(prior.Σ\prior.μ))
    ϑ = rand(Gaussian(μ_post, Σ))
    move_to_proper_place(ϑ, θ, updtIdx), μ_post, Σ
end


function _pseudo_conjugate_draw(ϑ, μ, 𝓦, XX, PT, updtIdx)
    for X in XX
        Γ⁻¹ = hypo_a_inv(PT, X.tt[end], X.yy[end])
        for i in 1:length(X)-1
            φₜ = φ(updtIdx, X.tt[i], X.yy[i], PT)
            φᶜₜ = φᶜ(updtIdx, ϑ, X.tt[i], X.yy[i], PT)
            dt = X.tt[i+1] - X.tt[i]
            dy = nonhypo(PT, X.yy[i+1])-nonhypo(PT, X.yy[i])
            μ = μ + φₜ'*Γ⁻¹*dy - φₜ'*Γ⁻¹*φᶜₜ*dt
            𝓦 = 𝓦 + φₜ'*Γ⁻¹*φₜ*dt
        end
    end
    μ, 𝓦
end
