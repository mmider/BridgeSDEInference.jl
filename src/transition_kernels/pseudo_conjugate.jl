
#@generated function thetainc(::Val{T}, θ) where T
#    z = Expr(:tuple, (:(θ[$i]) for i in 1:length(T) if  T[i])...)
#    return z
#end

function pseudo_conjugate_draw(θ, XX, PT, prior, updtIdx, α = 1.0)
    𝓦 = mustart(updtIdx)*mustart(updtIdx)'
    𝓦 = _pseudo_conjugate_draw(𝓦, XX, PT, updtIdx)

    Σ = inv(𝓦 + inv(Matrix(prior.Σ)))
    Σ = (Σ + Σ')/2 # eliminates numerical inconsistencies
    ϑ = thetainc(updtIdx, θ)
    ϑᵒ = rand(Gaussian(ϑ, α * Σ))
    move_to_proper_place(ϑᵒ, θ, updtIdx), Σ
end


function _pseudo_conjugate_draw(𝓦, XX, PT, updtIdx)
    for X in XX
        Γ⁻¹ = hypo_a_inv(PT, X.tt[end], X.yy[end])
        for i in 1:length(X)-1
            φₜ = φ(updtIdx, X.tt[i], X.yy[i], PT)
            dt = X.tt[i+1] - X.tt[i]
            𝓦 = 𝓦 + φₜ'*Γ⁻¹*φₜ*dt
        end
    end
    𝓦
end
