import Bridge: _scale, endpoint, samplepath, EulerMaruyama, ProcessOrCoefficients, _b#, SamplePath
using StaticArrays
"""
    forcedSolve!(::EulerMaruyama, Y, u::T, W::SamplePath,
                 P::ProcessOrCoefficients, ::DiffusionDomain=domain(P))

Solve stochastic differential equation ``dX_t = b(t,X_t)dt + σ(t,X_t)dW_t``
using the Euler-Maruyama scheme in place. `forcedSolve!` as opposed to `solve!`
enforces adherence to the diffusion's domain (which numerical schemes are prone
to violate). By default, no restrictions are made, so calls `solve!`.
"""
function forcedSolve!(::EulerMaruyama, Y, u::T, W::SamplePath,
                      P::ProcessOrCoefficients, ::UnboundedDomain=domain(P)
                      ) where {T}
    solve!(Euler(), Y, u, W, P)
end

"""
    forcedSolve!(::EulerMaruyama, Y, u::T,
                 W::Union{SamplePath{SVector{D,S}},SamplePath{S}},
                 P::ProcessOrCoefficients, d::LowerBoundedDomain=domain(P))

Solve stochastic differential equation ``dX_t = b(t,X_t)dt + σ(t,X_t)dW_t``
using the Euler-Maruyama scheme in place. `forcedSolve!` as opposed to `solve!`
enforces adherence to the diffusion's domain (which numerical schemes are prone
to violate). This function enforces lower bounds by modifying `W` in place.
"""
function forcedSolve!(::EulerMaruyama, Y, u::T,
                      W::Union{SamplePath{SVector{D,S}},SamplePath{S}},
                      P::ProcessOrCoefficients, d::LowerBoundedDomain=domain(P)
                      ) where {D,S,T}
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


function forcedSolve(::EulerMaruyama, u::T, W::SamplePath,
                     P::ProcessOrCoefficients) where T
    WW = deepcopy(W)
    XX = samplepath(W.tt, zero(T))
    forcedSolve!(Euler(), XX, u, WW, P)
    WW, XX
end
