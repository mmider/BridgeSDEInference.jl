import Bridge: ProcessOrCoefficients

function invSolve!(::EulerMaruyama, Y, W::SamplePath, P::ProcessOrCoefficients) where {T}
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt
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
