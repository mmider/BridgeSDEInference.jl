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

struct LMatrix <: ODEElement end
struct M⁺Matrix <: ODEElement end
struct μVector <: ODEElement end

"""
    update(::HMatrix, t, H, Hν, c, P)

ODE satisfied by `H`, i.e. d`H` = `update`(...)dt
"""
update(::HMatrix, t, H, Hν, c, P) = ( - Bridge.B(t, P)'*H - H*Bridge.B(t, P)
                                         + outer(H * Bridge.σ(t, P)) )
"""
    update(::HνVector, t, H, Hν, c, P)

ODE satisfied by `Hν`, i.e. d`Hν` = `update`(...)dt
"""
update(::HνVector, t, H, Hν, c, P) = ( - Bridge.B(t, P)'*Hν + H*a(t,P)*Hν
                                          + H*Bridge.β(t, P) )
"""
    update(::cScalar, t, H, Hν, c, P)

ODE satisfied by `c`, i.e. d`c` = `update`(...)dt
"""
update(::cScalar, t, H, Hν, c, P) = ( dot(Bridge.β(t, P), Hν)
                                         + 0.5*outer(Hν' * Bridge.σ(t, P))
                                         - 0.5*tr(H * a(t, P)))

"""
    update(::LMatrix, t, L, M⁺, μ, P)

ODE satisfied by `L`, i.e. d`L` = `update`(...)dt
"""
update(::LMatrix, t, L, M⁺, μ, P) = - L*Bridge.B(t, P)

"""
    update(::M⁺Matrix, t, L, M⁺, μ, P)

ODE satisfied by `M⁺`, i.e. d`M⁺` = `update`(...)dt
"""
update(::M⁺Matrix, t, L, M⁺, μ, P) = - outer(L * Bridge.σ(t, P))

"""
    update(::μVector, t, L, M⁺, μ, P)

ODE satisfied by `μ`, i.e. d`μ` = `update`(...)dt
"""
update(::μVector, t, L, M⁺, μ, P) = - L * Bridge.β(t, P)


createTableau(::T) where T = nothing
createTableau(::Tsit5) = Tsit5Tableau()
createTableau(::Vern7) = Vern7Tableau()

function reserveMemLM⁺μ(changePt::ODEChangePt, ::TH, ::THν) where {TH,THν}
    N = getChangePt(changePt)
    L̃ = zeros(TH, N) # NOTE: not TL
    M̃⁺ = zeros(TH, N) # NOTE: not TΣ
    μ = zeros(THν, N) # NOTE: not Tv
    L̃, M̃⁺, μ
end

function initLM⁺μ!(::NoChangePt, ::Any, ::Any, ::Any, ::Any, ::Any) end

function initLM⁺μ!(::ODEChangePt, L̃::Vector{TL}, M̃⁺::Vector{TΣ}, μ::Vector{Tμ},
                   L::TL, Σ::TΣ) where {TL,TΣ,Tμ}
    L̃[end] = L
    M̃⁺[end] = Σ
    μ[end] = zero(Tμ)
end

function HHνcFromLM⁺μ!(H, Hν, c, L̃, M̃⁺, μ, v, λ)
    N = length(H)
    d, d = size(M̃⁺[end])
    for i in λ:-1:1
        H[N-λ+i] = (L̃[i])' * (M̃⁺[i] \ L̃[i])
        Hν[N-λ+i] = (L̃[i])' * (M̃⁺[i] \ (v-μ[i]))
        c[N-λ+i] = ( 0.5 * (v - μ[i])' * (M̃⁺[i] \ (v - μ[i]))
                 - 0.5*d*log(2*π) - 0.5*log(det(M̃⁺[i])) )
    end
end


"""
    gpupdate!(t, L, Σ, v, H⁽ᵀ⁺⁾, Hν⁽ᵀ⁺⁾, c⁽ᵀ⁺⁾, H, Hν, c, P,
              solver::ST = Ralston3(), changePt::ODEChangePt)

Compute the values of elements `H`, `Hν`, `c`, on a grid of time-points.
...
# Arguments
- `t`: vector of time-points
- `L`: observation operator at the end-point
- `Σ`: covariance matrix of the noise perturbating observation
- `v`: observation at the end-point (`v` = `L`X + 𝓝(0,`Σ`))
- `H⁽ᵀ⁺⁾`: `H` at the left limit of subsequent interval
- `Hν⁽ᵀ⁺⁾`: `Hν` at the left limit of subsequent interval
- `c⁽ᵀ⁺⁾`: `c` at the left limit of subsequent interval
- `H`: container where values of `H` evaluated on a grid will be stored
- `Hν`: container where values of `Hν` evaluated on a grid will be stored
- `c`: container where values of `c` evaluated on a grid will be stored
- `P`: Law of a proposal diffusion
- `solver`: numerical solver used for solving the backward ODEs
...
"""
function gpupdate!(t, L, Σ, v, H⁽ᵀ⁺⁾, Hν⁽ᵀ⁺⁾, c⁽ᵀ⁺⁾, H, Hν, c, L̃, M̃⁺, μ, P,
                   changePt::ODEChangePt, solver::ST = Ralston3()) where ST
    m, d = size(L)
    @assert size(L[:,1]) == (m,)
    @assert size(L*L') == size(Σ) == (m, m)

    λ = _gpupdate!(changePt, t, L, Σ, v, H, Hν, c, L̃, M̃⁺, μ, P, ST())

    toUpdate = (HMatrix(), HνVector(), cScalar())
    tableau = createTableau(ST())

    N = length(t)
    for i in N-λ-1:-1:1
        dt = t[i] - t[i+1]
        H[i], Hν[i], c[i] = update(ST(), toUpdate, t[i+1], H[i+1], Hν[i+1],
                                   c[i+1], dt, P, tableau)
    end
end


function _gpupdate!(changePt::ODEChangePt, t, L, Σ, v, H, Hν, c, L̃, M̃⁺, μ, P,
                    solver::ST = Ralston3()) where ST
    toUpdate = (LMatrix(), M⁺Matrix(), μVector())
    λ = getChangePt(changePt)
    N = length(t)
    tableau = createTableau(ST())#solver(changePt))

    initLM⁺μ!(changePt, L̃, M̃⁺, μ, L, Σ)

    for i in λ-1:-1:1
        dt = t[N-λ+i] - t[N-λ+i+1]
        L̃[i], M̃⁺[i], μ[i] = update(ST(), toUpdate, t[N-λ+i+1], L̃[i+1], M̃⁺[i+1],
                                   μ[i+1], dt, P, tableau)
    end

    HHνcFromLM⁺μ!(H, Hν, c, L̃, M̃⁺, μ, v, λ)
    λ
end

function _gpupdate!(::NoChangePt, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any,
                    ::Any, ::Any, ::Any, ::Any, ::Any, ::Any)
    0
end



"""
     gpupdate!(P, H⁽ᵀ⁺⁾, Hν⁽ᵀ⁺⁾, c⁽ᵀ⁺⁾ solver::ST = Ralston3())

Re-compute the values of `H`, `Hν`, `c` on a grid of time-points. This
function is used by the mcmc sampler.
"""
function gpupdate!(P, H⁽ᵀ⁺⁾ = zero(typeof(P.H[1])),
                   Hν⁽ᵀ⁺⁾ = zero(typeof(P.Hν[1])), c⁽ᵀ⁺⁾ = 0.0;
                   solver::ST = Ralston3) where ST
    gpupdate!(P.tt, P.L, P.Σ, P.v, H⁽ᵀ⁺⁾, Hν⁽ᵀ⁺⁾, c⁽ᵀ⁺⁾, P.H,
              P.Hν, P.c, P.L̃, P.M̃⁺, P.μ, P.Pt, P.changePt, ST())
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
    L̃::Vector{S1}       # (optional) matrix L evaluated at time-points `tt`
    M̃⁺::Vector{TH}      # (optional) matrix M⁺ evaluated at time-points `tt`
    μ::Vector{Tν}       # (optional) vector μ evaluated at time-points `tt`
    L::S1               # observation operator (for observation at the end-pt)
    v::S2               # observation at the end-point
    Σ::S3               # covariance matrix of the noise at observation
    changePt::TC        # Info about the change point between ODE solvers
end
```
stores all information that is necessary for drawing guided proposals.

    GuidPropBridge(tt_, P, Pt, L::S1, v::S2, Σ::S3 = Bridge.outer(zero(v)),
                   H⁽ᵀ⁺⁾::TH = zero(typeof(L'*L)),
                   Hν⁽ᵀ⁺⁾::Tν = zero(typeof(L'[:,1])), c⁽ᵀ⁺⁾ = 0.0;
                   # H⁻¹prot is currently not used
                   H⁻¹prot::TH⁻¹ = SVector{prod(size(TH))}(rand(prod(size(TH)))),
                   solver::ST = Ralston3())

Base constructor that takes values of `H`, `Hν`, `c` and `Q` evaluated at the
left limit of the subsequent interval (given respectively by elements: `H⁽ᵀ⁺⁾`,
`Hν⁽ᵀ⁺⁾` and `c⁽ᵀ⁺⁾`) and automatically computes the elements `H`,
`Hν` and `c` for a given interval.

    GuidPropBridge(P::GuidPropBridge{T,R,R2,Tν,TH,TH⁻¹,S1,S2,S3}, θ)

Clone constructor. It creates a new object `GuidPropBridge` from the old one `P`
by using all internal containers of `P` and only defining new pointers that
point to the old memory locations. Additionally, `P.Target` and `P.Pt` are
deleted and substituted with their clones that use different value of parameter
`θ`.

    GuidPropBridge(P::GuidPropBridge{T,K,R,R2,Tν,TH,TH⁻¹,S̃1,S̃2,S̃3}, L::S1,
                   v::S2, Σ::S3, θ)

Another clone constructor. It creates a new object `GuidPropBridge` from the old
one `P` by using all internal containers of `P` and only defining new pointers
that point to the old memory locations. `P.Target` and `P.Pt` are deleted
and substituted with their clones that use different value of parameter `θ`.
Additionally, the observational operator `L`, covariance of the additive noise
at the observation time `Σ`, as well as the observation `v`  are all changed.
"""
struct GuidPropBridge{T,K,R,R2,Tν,TH,TH⁻¹,S1,S2,S3,TC} <: ContinuousTimeProcess{T}
    Target::R           # Law of the target diffusion
    Pt::R2              # Law of the proposal diffusion
    tt::Vector{Float64} # grid of time points
    H::Vector{TH}       # Matrix H evaluated at time-points `tt`
    H⁻¹::Vector{TH⁻¹}   # currently not used
    Hν::Vector{Tν}      # Vector Hν evaluated at time-points `tt`
    c::Vector{K}        # scalar c evaluated at time-points `tt`
    L̃::Vector{TH}       # (optional) matrix L evaluated at time-points `tt` NOTE not S1
    M̃⁺::Vector{TH}      # (optional) matrix M⁺ evaluated at time-points `tt` NOTE not S3
    μ::Vector{Tν}      # (optional) vector μ evaluated at time-points `tt` NOTE not S2
    L::S1               # observation operator (for observation at the end-pt)
    v::S2               # observation at the end-point
    Σ::S3               # covariance matrix of the noise at observation
    changePt::TC        # Info about the change point between ODE solvers

    function GuidPropBridge(::Type{K}, tt_, P, Pt, L::S1, v::S2,
                            Σ::S3 = Bridge.outer(zero(K)*zero(v)),
                            H⁽ᵀ⁺⁾::TH = zero(typeof(zero(K)*L'*L)),
                            Hν⁽ᵀ⁺⁾::Tν = zero(typeof(zero(K)*L'[:,1])),
                            c⁽ᵀ⁺⁾ = zero(K);
                            # H⁻¹prot is currently not used
                            H⁻¹prot::TH⁻¹ = SVector{prod(size(TH))}(rand(prod(size(TH)))),
                            changePt::TC = NoChangePt(),
                            solver::ST = Ralston3()
                            ) where {K,Tν,TH,TH⁻¹,S1,S2,S3,ST,TC}
        tt = collect(tt_)
        N = length(tt)
        H = zeros(TH, N)
        H⁻¹ = zeros(TH⁻¹, N)
        Hν = zeros(Tν, N)
        c = zeros(K, N)

        L̃, M̃⁺, μ = reserveMemLM⁺μ(changePt, H[1], Hν[1])

        gpupdate!(tt, L, Σ, v, H⁽ᵀ⁺⁾, Hν⁽ᵀ⁺⁾, c⁽ᵀ⁺⁾, H, Hν, c, L̃, M̃⁺, μ, Pt,
                  changePt, ST())

        T = Bridge.valtype(P)
        R = typeof(P)
        R2 = typeof(Pt)

        new{T,K,R,R2,Tν,TH,TH⁻¹,S1,S2,S3,TC}(P, Pt, tt, H, H⁻¹, Hν, c, L̃, M̃⁺, μ,
                                             L, v, Σ, changePt)
    end

    function GuidPropBridge(P::GuidPropBridge{T,K,R,R2,Tν,TH,TH⁻¹,S1,S2,S3,TC},
                            θ) where {T,K,R,R2,Tν,TH,TH⁻¹,S1,S2,S3,TC}
        new{T,K,R,R2,Tν,TH,TH⁻¹,S1,S2,S3,TC}(clone(P.Target,θ), clone(P.Pt,θ),
                                             P.tt, P.H, P.H⁻¹, P.Hν, P.c, P.L̃,
                                             P.M̃⁺, P.μ, P.L, P.v, P.Σ,
                                             P.changePt)
    end

    function GuidPropBridge(P::GuidPropBridge{T,K,R,R2,Tν,TH,TH⁻¹,S̃1,S̃2,S̃3,TC̃},
                            L::S1, v::S2, Σ::S3, changePt::TC, θ
                            ) where {T,K,R,R2,Tν,TH,TH⁻¹,S̃1,S̃2,S̃3,S1,S2,S3,TC̃,TC}
        new{T,K,R,R2,Tν,TH,TH⁻¹,S1,S2,S3,TC}(clone(P.Target,θ), clone(P.Pt,θ),
                                             P.tt, P.H, P.H⁻¹, P.Hν, P.c, P.L̃,
                                             P.M̃⁺, P.μ, L, v, Σ, changePt)
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
    - 0.5 * ( x₀'*P.H[1]*x₀ - 2.0*dot(P.Hν[1], x₀) ) - P.c[1]
end
