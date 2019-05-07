using Bridge, LinearAlgebra, StaticArrays
import Bridge: IndexedTime, outer, _b, r, H, σ, a, Γ, constdiff, b
import Bridge: target, auxiliary
import Base: valtype

"""
    ODEElement

Types inheriting from abstract type `ODEElement` are used to differentiate
at-compilation-time between the appropriate sets of ODEs to be used
"""
abstract type ODEElement end
struct HMatrix <: ODEElement end
struct HνVector <: ODEElement end
struct cScalar <: ODEElement end
struct QScalar <: ODEElement end

"""
    update(::HMatrix, t, H, Hν, c, Q, P)

ODE satisfied by `H`, i.e. d`H` = `update`(...)dt
"""
update(::HMatrix, t, H, Hν, c, Q, P) = ( - Bridge.B(t, P)'*H - H*Bridge.B(t, P)
                                         + outer(H * Bridge.σ(t, P)) )
"""
    update(::HνVector, t, H, Hν, c, Q, P)

ODE satisfied by `Hν`, i.e. d`Hν` = `update`(...)dt
"""
update(::HνVector, t, H, Hν, c, Q, P) = ( - Bridge.B(t, P)'*Hν + H*a(t,P)*Hν
                                          + H*Bridge.β(t, P) )
"""
    update(::cScalar, t, H, Hν, c, Q, P)

ODE satisfied by `c`, i.e. d`c` = `update`(...)dt
"""
update(::cScalar, t, H, Hν, c, Q, P) = ( 2.0*dot(Bridge.β(t, P), Hν)
                                         + outer(Hν' * Bridge.σ(t, P)) )
"""
    update(::QScalar, t, H, Hν, c, Q, P)

ODE satisfied by `Q`, i.e. d`Q` = `update`(...)dt
"""
update(::QScalar, t, H, Hν, c, Q, P) = -0.5*tr(H * a(t, P))

createTableau(::T) where T = nothing
createTableau(::Tsit5) = Tsit5Tableau()
createTableau(::Vern7) = Vern7Tableau()

"""
    gpupdate!(t, L, Σ, v, H⁽ᵀ⁺⁾, Hν⁽ᵀ⁺⁾, c⁽ᵀ⁺⁾, Q⁽ᵀ⁺⁾, H, Hν, c, Q, P,
              solver::ST = Ralston3())

Compute the values of elements `H`, `Hν`, `c`, `Q` on a grid of
time-points.
...
# Arguments
- `t`: vector of time-points
- `L`: observation operator at the end-point
- `Σ`: covariance matrix of the noise perturbating observation
- `v`: observation at the end-point (`v` = `L`X + 𝓝(0,`Σ`))
- `H⁽ᵀ⁺⁾`: `H` at the left limit of subsequent interval
- `Hν⁽ᵀ⁺⁾`: `Hν` at the left limit of subsequent interval
- `c⁽ᵀ⁺⁾`: `c` at the left limit of subsequent interval
- `Q⁽ᵀ⁺⁾`: `Q` at the left limit of subsequent interval
- `H`: container where values of `H` evaluated on a grid will be stored
- `Hν`: container where values of `Hν` evaluated on a grid will be stored
- `c`: container where values of `c` evaluated on a grid will be stored
- `Q`: container where values of `Q` evaluated on a grid will be stored
- `P`: Law of a proposal diffusion
- `solver`: numerical solver used for solving the backward ODEs
...
"""
function gpupdate!(t, L, Σ, v, H⁽ᵀ⁺⁾, Hν⁽ᵀ⁺⁾, c⁽ᵀ⁺⁾, Q⁽ᵀ⁺⁾, H, Hν, c,
                   Q, P, solver::ST = Ralston3()) where ST
    m, d = size(L)
    @assert size(L[:,1]) == (m,)
    @assert size(L*L') == size(Σ) == (m, m)

    toUpdate = (HMatrix(), HνVector(), cScalar(), QScalar())
    tableau = createTableau(ST())

    H[end] = H⁽ᵀ⁺⁾ + L' * (Σ \ L)
    Hν[end] = Hν⁽ᵀ⁺⁾ + L' * (Σ \ v)
    c[end] = c⁽ᵀ⁺⁾ + v' * (Σ \ v)
    Q[end] = Q⁽ᵀ⁺⁾ + 0.5*m*log(2.0*π) + 0.5*log(abs(det(Σ)))

    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        H[i], Hν[i], c[i], Q[i] = update(ST(), toUpdate, t[i+1], H[i+1],
                                         Hν[i+1], c[i+1], Q[i+1], dt, P,
                                         tableau)
    end
end

"""
     gpupdate!(P, H⁽ᵀ⁺⁾, Hν⁽ᵀ⁺⁾, c⁽ᵀ⁺⁾, Q⁽ᵀ⁺⁾, solver::ST = Ralston3())

Re-compute the values of `H`, `Hν`, `c`, `Q` on a grid of time-points. This
function is used by the mcmc sampler.
"""
function gpupdate!(P, H⁽ᵀ⁺⁾ = zero(typeof(P.H[1])),
                   Hν⁽ᵀ⁺⁾ = zero(typeof(P.Hν[1])), c⁽ᵀ⁺⁾ = 0.0, Q⁽ᵀ⁺⁾ = 0.0;
                   solver::ST = Ralston3) where ST
    gpupdate!(P.tt, P.L, P.Σ, P.v, H⁽ᵀ⁺⁾, Hν⁽ᵀ⁺⁾, c⁽ᵀ⁺⁾, Q⁽ᵀ⁺⁾, P.H,
              P.Hν, P.c, P.Q, P.Pt, ST())
end


"""
    GuidPropBridge

Struct
```
struct GuidPropBridge{T,R,R2,Tν,TH,TH⁻¹,S1,S2,S3} <: ContinuousTimeProcess{T}
    Target::R           # Law of the target diffusion
    Pt::R2              # Law of the proposal diffusion
    tt::Vector{Float64} # grid of time points
    H::Vector{TH}       # Matrix H evaluated at time-points `tt`
    H⁻¹::Vector{TH⁻¹}   # currently not used
    Hν::Vector{Tν}      # Vector Hν evaluated at time-points `tt`
    c::Vector{Float64}  # scalar c evaluated at time-points `tt`
    Q::Vector{Float64}  # scalar Q evaluated at time-points `tt`
    L::S1               # observation operator (for observation at the end-pt)
    v::S2               # observation at the end-point
    Σ::S3               # covariance matrix of the noise at observation
end
```
stores all information that is necessary for drawing guided proposals.

    GuidPropBridge(tt_, P, Pt, L::S1, v::S2, Σ::S3 = Bridge.outer(zero(v)),
                   H⁽ᵀ⁺⁾::TH = zero(typeof(L'*L)),
                   Hν⁽ᵀ⁺⁾::Tν = zero(typeof(L'[:,1])), c⁽ᵀ⁺⁾ = 0.0, Q⁽ᵀ⁺⁾ = 0.0;
                   # H⁻¹prot is currently not used
                   H⁻¹prot::TH⁻¹ = SVector{prod(size(TH))}(rand(prod(size(TH)))),
                   solver::ST = Ralston3())

Base constructor that takes values of `H`, `Hν`, `c` and `Q` evaluated at the
left limit of the subsequent interval (given respectively by elements: `H⁽ᵀ⁺⁾`,
`Hν⁽ᵀ⁺⁾`, `c⁽ᵀ⁺⁾` and `Q⁽ᵀ⁺⁾`) and automatically computes the elements `H`,
`Hν`, `c` and `Q` for a given interval.

    GuidPropBridge(P::GuidPropBridge{T,R,R2,Tν,TH,TH⁻¹,S1,S2,S3}, θ)

Clone constructor. It creates a new object `GuidPropBridge` from the old one `P`
by using all internal containers of `P` and only defining new pointers that
point to the old memory locations. Additionally, `P.Target` and `P.Pt` are
deleted and substituted with their clones that use different value of parameter
`θ`.
"""
struct GuidPropBridge{T,K,R,R2,Tν,TH,TH⁻¹,S1,S2,S3} <: ContinuousTimeProcess{T}
    Target::R           # Law of the target diffusion
    Pt::R2              # Law of the proposal diffusion
    tt::Vector{Float64} # grid of time points
    H::Vector{TH}       # Matrix H evaluated at time-points `tt`
    H⁻¹::Vector{TH⁻¹}   # currently not used
    Hν::Vector{Tν}      # Vector Hν evaluated at time-points `tt`
    c::Vector{K}        # scalar c evaluated at time-points `tt`
    Q::Vector{K}        # scalar Q evaluated at time-points `tt`
    L::S1               # observation operator (for observation at the end-pt)
    v::S2               # observation at the end-point
    Σ::S3               # covariance matrix of the noise at observation

    function GuidPropBridge(::Type{K}, tt_, P, Pt, L::S1, v::S2,
                            Σ::S3 = Bridge.outer(zero(K)*zero(v)),
                            H⁽ᵀ⁺⁾::TH = zero(typeof(zero(K)*L'*L)),
                            Hν⁽ᵀ⁺⁾::Tν = zero(typeof(zero(K)*L'[:,1])),
                            c⁽ᵀ⁺⁾ = zero(K),
                            Q⁽ᵀ⁺⁾ = zero(K);
                            # H⁻¹prot is currently not used
                            H⁻¹prot::TH⁻¹ = SVector{prod(size(TH))}(rand(prod(size(TH)))),
                            solver::ST = Ralston3()
                            ) where {K,Tν,TH,TH⁻¹,S1,S2,S3,ST}
        tt = collect(tt_)
        N = length(tt)
        H = zeros(TH, N)
        H⁻¹ = zeros(TH⁻¹, N)
        Hν = zeros(Tν, N)
        c = zeros(K, N)
        Q = zeros(K, N)

        gpupdate!(tt, L, Σ, v, H⁽ᵀ⁺⁾, Hν⁽ᵀ⁺⁾, c⁽ᵀ⁺⁾, Q⁽ᵀ⁺⁾, H, Hν, c, Q,
                  Pt, ST())

        T = Bridge.valtype(P)
        R = typeof(P)
        R2 = typeof(Pt)
        new{T,K,R,R2,Tν,TH,TH⁻¹,S1,S2,S3}(P, Pt, tt, H, H⁻¹, Hν, c, Q, L, v, Σ)
    end

    function GuidPropBridge(P::GuidPropBridge{T,K,R,R2,Tν,TH,TH⁻¹,S1,S2,S3},
                            θ) where {T,K,R,R2,Tν,TH,TH⁻¹,S1,S2,S3}
        new{T,K,R,R2,Tν,TH,TH⁻¹,S1,S2,S3}(clone(P.Target,θ), clone(P.Pt,θ), P.tt,
                                        P.H, P.H⁻¹, P.Hν, P.c, P.Q, P.L, P.v,
                                        P.Σ)
    end

    function GuidPropBridge(P::GuidPropBridge{T,K,R,R2,Tν,TH,TH⁻¹,S̃1,S̃2,S̃3},
                            L::S1, v::S2,
                            Σ::S3) where {T,K,R,R2,Tν,TH,TH⁻¹,S̃1,S̃2,S̃3,S1,S2,S3}
        new{T,K,R,R2,Tν,TH,TH⁻¹,S1,S2,S3}(P.Target, P.Pt, P.tt, P.H, P.H⁻¹,
                                          P.Hν, P.c, P.Q, L, v, Σ)
    end
end


function _b((i,t)::IndexedTime, x, P::GuidPropBridge)
    b(P.tt[i], x, P.Target) + a(P.tt[i], x, P.Target)*(P.Hν[i]-P.H[i]*x)
end

r((i,t)::IndexedTime, x, P::GuidPropBridge) = P.Hν[i]-P.H[i]*x
H((i,t)::IndexedTime, x, P::GuidPropBridge) = P.H[i]


σ(t, x, P::GuidPropBridge) = σ(t, x, P.Target)
a(t, x, P::GuidPropBridge) = a(t, x, P.Target)
Γ(t, x, P::GuidPropBridge) = Γ(t, x, P.Target)
constdiff(P::GuidPropBridge) = constdiff(P.Target) && constdiff(P.Pt)


"""
    llikelihood(::LeftRule, X::SamplePath, P::GuidPropBridge; skip = 0)

Log-likelihood for the imputed path `X` under the target law `P.Target` with
respect to the proposal law `P.Pt`. Use Riemann sum approximation to an
integral, evaluating f(xᵢ), i=1,… at the left limit of intervals and skipping
`skip` many points between each evaluation of f(xᵢ) for efficiency.
"""
function llikelihood(::LeftRule, X::SamplePath, P::GuidPropBridge; skip = 0)
    tt = X.tt
    xx = X.yy
    som = 0.0 # hopefully this instability gets optimised away
    for i in 1:length(tt)-1-skip #skip last value, summing over n-1 elements
        s = tt[i]
        x = xx[i]
        r = Bridge.r((i,s), x, P)

        som += ( dot( _b((i,s), x, target(P)) - _b((i,s), x, auxiliary(P)), r )
                 * (tt[i+1]-tt[i]) )

        if !constdiff(P)
            H = H((i,s), x, P)
            som -= ( 0.5*tr( (a((i,s), x, target(P))
                             - aitilde((i,s), x, P))*H ) * (tt[i+1]-tt[i]) )
            som += ( 0.5*( r'*(a((i,s), x, target(P))
                           - aitilde((i,s), x, P))*r ) * (tt[i+1]-tt[i]) )
        end
    end
    som
end

"""
    lobslikelihood(P::GuidPropBridge, x₀)

Log-likelihood for the observations under the auxiliary law, for a diffusion
started from x₀.
"""
function lobslikelihood(P::GuidPropBridge, x₀)
    - 0.5 * ( x₀'*P.H[1]*x₀ - 2.0*dot(P.Hν[1], x₀) + P.c[1] ) - P.Q[1]
end
