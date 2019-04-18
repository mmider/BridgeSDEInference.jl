"""
    conjugateDraw(θ, XX, P, prior, ::updtIdx)

Draw from the full conditional distribution of the parameters whose indices are
specified by the object `updtIdx`, conditionally on the path given in container
`XX`, and conditionally on all other parameter values given in vector `θ`.
"""
function conjugateDraw(θ, XX, P, prior, updtIdx)
    μ = mustart(updtIdx)
    𝓦 = μ*μ'
    PT = P[1].Target
    ϑ = SVector(thetaex(updtIdx, θ))
    μ, 𝓦 = _conjugateDraw(ϑ, μ, 𝓦, XX, PT, updtIdx)

    Σ = inv(𝓦 + inv(Matrix(prior.Σ)))
    Σ = (Σ + Σ')/2 # eliminates numerical inconsistencies
    μₚₒₛₜ = Σ * (μ + Vector(prior.Σ\prior.μ))
    rand(MvNormal(μₚₒₛₜ, Matrix{Float64}(Σ)))
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
