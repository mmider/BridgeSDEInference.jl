import Bridge: ProcessOrCoefficients


"""
    invSolve!(::EulerMaruyama, Y, W::SamplePath, P::ProcessOrCoefficients)

Compute Wiener process `W` that would have been needed to be simulated in order
to obtain path `Y` under law `P` when using Euler-Maruyama numerical scheme
"""
function inv_solve!(::EulerMaruyama, Y, W::SamplePath, P::ProcessOrCoefficients
                  ) where {T}
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = W.tt

    yy = Y.yy
    #tt[:] = Y.tt
    ww[.., N] = zero(ww[.., N])

    for i in N-1:-1:1
        yᵢ₊₁ = yy[.., i+1]
        yᵢ = yy[.., i]

        ww[.., i] = ww[.., i+1] - σ(tt[i], yᵢ, P)\(yᵢ₊₁ - yᵢ - _b((i,tt[i]), yᵢ, P)*(tt[i+1]-tt[i]))
    end
    for i in N:-1:1
        ww[.., i] = ww[.., i] - ww[.., 1]
    end
    W
end
