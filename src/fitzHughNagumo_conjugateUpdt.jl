using GaussianDistributions


"""
    φ(::Val{T}, args...)

Compute the φ function appearing in the Girsanov formula and needed for
sampling from the full conditional distribution of the parameters (whose
indices are specified by the `Val`) conditional on the path,
observations and other parameters.
"""
@generated function φ(::Val{T}, args...) where T
    z = Expr(:tuple, (:(phi(Val($i), args...)) for i in 1:length(T) if T[i])...)
    return z
end

"""
    φᶜ(::Val{T}, args...)

Compute the φᶜ function appearing in the Girsanov formula. This function
complements φ.
"""
@generated function φᶜ(::Val{T}, args...) where T
    z = Expr(:tuple, (:(phi(Val($i), args...)) for i in 0:length(T) if i==0 || !T[i])...)
    return z
end

phi(::Val{0}, t, x, P::FitzhughDiffusion) = -x[2]
phi(::Val{1}, t, x, P::FitzhughDiffusion) = x[1]-x[1]^3+(1-3*x[1]^2)*x[2]
phi(::Val{2}, t, x, P::FitzhughDiffusion) = one(x[1])
phi(::Val{3}, t, x, P::FitzhughDiffusion) = -x[1]
phi(::Val{4}, t, x, P::FitzhughDiffusion) = zero(x[1])
phi(::Val{5}, t, x, P::FitzhughDiffusion) = zero(x[1])


"""
    conjugateDraw(θ, XX, PT, prior, ::updtIdx)

Draw from the full conditional distribution of the parameters whose indices are
specified by the object `updtIdx`, conditionally on the path given in container
`XX`, and conditionally on all other parameter values given in vector `θ`.
"""
function conjugateDraw(θ, XX, PT, prior, updtIdx)
    μ = mustart(updtIdx)
    𝓦 = μ*μ'
    ϑ = SVector(thetaex(updtIdx, θ))
    μ, 𝓦 = _conjugateDraw(ϑ, μ, 𝓦, XX, PT, updtIdx)

    Σ = inv(𝓦 + inv(Matrix(prior.Σ)))
    Σ = (Σ + Σ')/2 # eliminates numerical inconsistencies
    μₚₒₛₜ = Σ * (μ + Vector(prior.Σ\prior.μ))
    rand(Gaussian(μₚₒₛₜ, Σ))
end
mustart(::Val{T}) where {T} = @SVector zeros(sum(T))
@generated function thetaex(::Val{T}, θ) where T
    z = Expr(:tuple, 1.0, (:(θ[$i]) for i in 1:length(T) if  !T[i])...)
    return z
end


function _conjugateDraw(ϑ, μ, 𝓦, XX, PT, updtIdx)
    for X in XX
        for i in 1:length(X)-1
            φₜ = SVector(φ(updtIdx, X.tt[i], X.yy[i], PT))
            φᶜₜ = SVector(φᶜ(updtIdx, X.tt[i], X.yy[i], PT))
            dt = X.tt[i+1] - X.tt[i]
            dy = X.yy[i+1][2]-X.yy[i][2]
            μ = μ + φₜ*dy - φₜ*dot(ϑ, φᶜₜ)*dt
            𝓦 = 𝓦 + φₜ*φₜ'*dt
        end
    end
    μ = μ/PT.σ^2
    𝓦 = 𝓦/PT.σ^2
    μ, 𝓦
end
