"""
    conjugateDraw(θ, XX, P, prior, ::updtIdx)

Draw from the full conditional distribution of the parameters whose indices are
specified by the object `updtIdx`, conditionally on the path given in container
`XX`, and conditionally on all other parameter values given in vector `θ`.
"""
function conjugateDraw(θ, XX, P, prior, ::updtIdx) where updtIdx
    n = length(idx(updtIdx()))
    𝓦 = zeros(n, n)
    μ = zeros(n)
    PT = P[1].Target
    temp = nonidx(updtIdx(), Val(3))
    ϑ = SVector{length(temp)+1}([1.0, temp...])

    for X in XX
        for i in 1:length(X)-1
            φₜ = SVector(φ(updtIdx(), X.tt[i], X.yy[i], PT))
            𝜙ₜ = SVector(𝜙(updtIdx(), X.tt[i], X.yy[i], PT))
            dt = X.tt[i+1] - X.tt[i]
            dy = X.yy[i+1][2]-X.yy[i][2]
            μ = μ + φₜ*dy - φₜ*dot(ϑ, 𝜙ₜ)*dt
            𝓦 = 𝓦 + φₜ*φₜ'*dt
        end
    end
    μ = μ/PT.σ^2
    𝓦 = 𝓦/PT.σ^2

    Σ = inv(𝓦 + inv(Matrix(prior.Σ)))
    Σ = (Σ + Σ')/2 # eliminates numerical inconsistencies
    μₚₒₛₜ = Σ * (μ + Vector(prior.Σ\prior.μ))
    rand(MvNormal(μₚₒₛₜ, Matrix{Float64}(Σ)))
end
