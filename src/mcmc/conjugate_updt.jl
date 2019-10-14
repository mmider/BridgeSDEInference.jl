using GaussianDistributions


@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = tuplejoin(tuplejoin(x, y), z...)


#NOTE it seems to be IMPERATIVE to define the function `num_non_hypo` PRIOR to
# running the @generated functions below due to the `world age problem`
# to this end in BridgeSDEInference all `examples` are imported before this file


"""
    φ(::Val{T}, args...)

Compute the φ function appearing in the Girsanov formula and needed for
sampling from the full conditional distribution of the parameters (whose
indices are specified by the `Val`) conditional on the path,
observations and other parameters.
"""
@generated function φ_old(::Val{T}, args...) where T
    z = Expr(:tuple, (:(phi(Val($i), args...)) for i in 1:length(T) if T[i])...)
    return z
end

@generated function φ(::Val{T}, t, x, P::S) where {T,S}
    data = Expr(:call, :tuplejoin, (:(phi(Val($i), t, x, P)) for i in 1:length(T) if T[i])...)
    mat = Expr(:call, SMatrix{num_non_hypo(S),sum(T)}, data)
    return mat
end


"""
    φᶜ(::Val{T}, args...)

Compute the φᶜ function appearing in the Girsanov formula. This function
complements φ.
"""
# by default define as a linear map
φᶜ(p, θ, args...) = φᶜlinear(p, args...) * θ

@generated function φᶜlinear(::Val{T}, t, x, P::S) where {T,S}
    data = Expr(:call, :tuplejoin, (:(phi(Val($i), t, x, P)) for i in 0:length(T) if i==0 || !T[i])...)
    mat = Expr(:call, SMatrix{num_non_hypo(S),length(T)-sum(T)+1}, data)
    return mat
end



"""
    conjugateDraw(θ, XX, PT, prior, ::updtIdx)

Draw from the full conditional distribution of the parameters whose indices are
specified by the object `updtIdx`, conditionally on the path given in container
`XX`, and conditionally on all other parameter values given in vector `θ`.
"""
function conjugate_draw(θ, XX, PT, prior, updtIdx)
    μ = mustart(updtIdx)
    𝓦 = μ*μ'
    ϑ = SVector(thetaex(updtIdx, θ))
    μ, 𝓦 = _conjugate_draw(ϑ, μ, 𝓦, XX, PT, updtIdx)

    Σ = inv(𝓦 + inv(Matrix(prior.Σ)))
    Σ = (Σ + Σ')/2 # eliminates numerical inconsistencies
    μ_post = Σ * (μ + Vector(prior.Σ\prior.μ))
    ϑ = rand(Gaussian(μ_post, Σ))
    move_to_proper_place(ϑ, θ, updtIdx)     # align so that dimensions agree
end


mustart(::Val{T}) where {T} = @SVector zeros(sum(T))
@generated function thetaex(::Val{T}, θ) where T
    z = Expr(:tuple, 1.0, (:(θ[$i]) for i in 1:length(T) if  !T[i])...)
    return z
end


function _conjugate_draw_old(ϑ, μ, 𝓦, XX, PT, updtIdx)
    for X in XX
        for i in 1:length(X)-1
            φₜ = SVector(φ(updtIdx, X.tt[i], X.yy[i], PT))
            φᶜₜ = SVector(φᶜ(updtIdx, X.tt[i], X.yy[i], PT))
            dt = X.tt[i+1] - X.tt[i]
            dy = X.yy[i+1][2]-X.yy[i][2]
            μ = μ + (φₜ*dy - φₜ*dot(ϑ, φᶜₜ)*dt)/PT.σ^2 #safe to use a(X.tt[i], X.yy[i], PT)
            𝓦 = 𝓦 + (φₜ*φₜ'*dt)/PT.σ^2
        end
    end
    μ, 𝓦
end

"""
    hypo_a_inv(P, t, x)

Base definition, assumes no hypoellipticity and no closed form expression for
the inverse of `a`
"""
hypo_a_inv(P, t, x) = inv(a(P, t, x))
nonhypo(::Any, x) = x

function _conjugate_draw(ϑ, μ, 𝓦, XX, PT, updtIdx)
    for X in XX
        for i in 1:length(X)-1
            φₜ = φ(updtIdx, X.tt[i], X.yy[i], PT)
            φᶜₜ = φᶜ(updtIdx, ϑ, X.tt[i], X.yy[i], PT)
            Γ⁻¹ = hypo_a_inv(PT, X.tt[i], X.yy[i])
            dt = X.tt[i+1] - X.tt[i]
            dy = nonhypo(PT, X.yy[i+1])-nonhypo(PT, X.yy[i])
            μ = μ + φₜ'*Γ⁻¹*dy - φₜ'*Γ⁻¹*φᶜₜ*dt
            𝓦 = 𝓦 + φₜ'*Γ⁻¹*φₜ*dt
        end
    end
    μ, 𝓦
end
