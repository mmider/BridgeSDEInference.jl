import Bridge: _scale, endpoint, samplepath, ProcessOrCoefficients, _b#, SamplePath
using StaticArrays


function forcedSolve!(em, Y, u, W::SamplePath,
                      P::ProcessOrCoefficients)
    forcedSolve!(em, Y, u, W, P, domain(P))
end

"""
    forcedSolve!(em, Y, u::T, W::SamplePath,
                 P::ProcessOrCoefficients, ::DiffusionDomain=domain(P))

Solve stochastic differential equation ``dX_t = b(t,X_t)dt + σ(t,X_t)dW_t``
using the Euler-Maruyama scheme in place. `forcedSolve!` as opposed to `solve!`
enforces adherence to the diffusion's domain (which numerical schemes are prone
to violate). By default, no restrictions are made, so calls `solve!`.
"""
function forcedSolve!(em::EulerMaruyamaBounded, Y, u::T, W::SamplePath,
                      P::ProcessOrCoefficients, ::UnboundedDomain) where {T}
    solve!(em, Y, u, W, P)
end

"""
    forcedSolve!(::EulerMaruyamaBounded, Y, u::T,
                 W::Union{SamplePath{SVector{D,S}},SamplePath{S}},
                 P::ProcessOrCoefficients, d::LowerBoundedDomain=domain(P))

Solve stochastic differential equation ``dX_t = b(t,X_t)dt + σ(t,X_t)dW_t``
using the Euler-Maruyama scheme in place. `forcedSolve!` as opposed to `solve!`
enforces adherence to the diffusion's domain (which numerical schemes are prone
to violate). This function enforces lower bounds by modifying `W` in place.
"""
function forcedSolve!(::EulerMaruyamaBounded, Y, u::T, W::SamplePath{S},
                      P::ProcessOrCoefficients, d::DiffusionDomain) where {S,T}
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
        dWt = ww[.., i+1]-ww[.., i]em
        increm = _b((i,tt[i]), y, P)*(tt[i+1]-tt[i]) + _scale(dWt, σ(tt[i], y, P))
        y_new = y + increm
        offset_addon = zero(S)
        while !bound_satisfied(d, y_new)
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


function forcedSolve(em, u::T, W::SamplePath,
                     P::ProcessOrCoefficients) where T
    WW = deepcopy(W)
    XX = samplepath(W.tt, zero(T))
    forcedSolve!(em, XX, u, WW, P)
    WW, XX
end

function forcedSolve(em, u::T, W::SamplePath,
                     P::GuidPropBridge) where T
    WW = deepcopy(W)
    XX = samplepath(W.tt, zero(T))
    forcedSolve!(em, XX, u, WW, P, domain(P.Target))
    WW, XX
end

import Bridge.solve!

function solve!(::Union{EulerMaruyamaBounded,Bridge.EulerMaruyama}, Y, u::T, W::SamplePath, P::ProcessOrCoefficients) where {T}
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt

    y::T = u
    dom = domain(P)
    for i in 1:N-1
        yy[.., i] = y
        y = y + _b((i,tt[i]), y, P)*(tt[i+1]-tt[i]) + _scale((ww[.., i+1]-ww[..,i]), σ(tt[i], y, P))
        !bound_satisfied(dom, y) && return false, nothing
    end
    yy[.., N] = endpoint(y, P)
    true, Y
end

#NOTE ↓↓↓↓↓↓↓↓↓↓↓ not the right way to go for now ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

function solveAndll(::EulerMaruyama, u::T, W::SamplePath, P::GuidPropBridge, θ
                    ) where T
    WW = deepcopy(W)
    XX = samplepath(W.tt, zero(T))
    ll = solveAndll!(EulerMaruyamaBounded(), XX, u, WW, P, domain(P.Target), θ)
    WW, XX, ll
end

function solveAndll!(::EulerMaruyama, u::T, W::SamplePath, P::GuidPropBridge, θ
                     ) where T
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt
    y::T = u

    ll = 0.0
    for i in 1:N-1
        yy[.., i] = y
        dWt = ww[.., i+1]-ww[.., i]
        s = tt[i]
        dt = tt[i+1]-tt[i]
        b_prop = _b((i,s), y, P, θ)
        y = y + b_prop*dt + _scale(dWt, σ(s, y, P, θ))

        b_trgt = _b((i,s), y, target(P), θ)
        b_aux = _b((i,s), y, auxiliary(P), θ)
        rₜₓ = r((i,s), y, P, θ)
        ll += dot(b_trgt-b_aux, rₜₓ) * dt

        if !constdiff(P)
            Hₜₓ = H((i,s), y, P, θ)
            aₜₓ = a((i,s), y, target(P), θ)
            ãₜ = ã((i,s), y, P, θ)
            ll -= 0.5*sum( (aₜₓ - ãₜ).*Hₜₓ ) * dt
            ll += 0.5*( rₜₓ'*(aₜₓ - ãₜ)*rₜₓ ) * dt
        end
    end
    yy[.., N] = endpoint(y, P)
    ll
end
