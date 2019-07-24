# NOTE parts are taken from Bridge.jl package: `https://github.com/mschauer/Bridge.jl`
# used files: `src/types.jl` and `src/euler.jl`
import Bridge: solve!, _scale, endpoint, EulerMaruyama, ProcessOrCoefficients, _b#, SamplePath
using StaticArrays
"""
    solve!(::EulerMaruyama, Y, u, W, P) -> X

Solve stochastic differential equation ``dX_t = b(t,X_t)dt + σ(t,X_t)dW_t``
using the Euler-Maruyama scheme in place.
"""
function solve!(::EulerMaruyama, Y, u::T, W::SamplePath,
                P::ProcessOrCoefficients, ::DiffusionDomain=domain(P)) where {T}
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    print("in there\n")
    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt
    y::T = u

    for i in 1:N-1
        yy[.., i] = y
        y = y + _b((i,tt[i]), y, P)*(tt[i+1]-tt[i]) + _scale((ww[.., i+1]-ww[..,i]), σ(tt[i], y, P))
    end
    yy[.., N] = endpoint(y, P)
    Y
end

"""
    solve!(::EulerMaruyama, Y, u, W, P) -> X

Solve stochastic differential equation ``dX_t = b(t,X_t)dt + σ(t,X_t)dW_t``
using the Euler-Maruyama scheme in place.
"""
function solve!(::EulerMaruyama, Y, u::T, W::Union{SamplePath{SVector{D,S}},SamplePath{S}},
                P::ProcessOrCoefficients,
                d::LowerBoundedDomain=domain(P)) where {D,S,T}
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt
    y::T = u

    offset = zero(S)
    for i in 1:N-1
        ww[..,i+1] += offset
        yy[.., i] = y
        dWt = ww[.., i+1]-ww[.., i]
        increm = _b((i,tt[i]), y, P)*(tt[i+1]-tt[i]) + _scale(dWt, σ(tt[i], y, P))
        y_new = y + increm
        offset_addon = zero(S)
        while !boundSatisfied(d, y_new)
            rootdt = √(tt[i+1]-tt[i])
            dWt = rootdt*randn(S)
            increm = _b((i,tt[i]), y, P)*(tt[i+1]-tt[i]) + _scale(dWt, σ(tt[i], y, P))
            y_new = y + increm
            offset_addon = ww[.., i] + dWt - ww[.., i+1]
        end
        offset += offset_addon
        ww[.., i+1] = ww[.., i] + dWt
        y = y_new
    end
    yy[.., N] = endpoint(y, P)
    Y
end
