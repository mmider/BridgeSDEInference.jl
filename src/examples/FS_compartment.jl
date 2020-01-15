using Bridge
#using Bridge.Models: ℝ
import Bridge: b, σ, B, β, a, constdiff

################## specify target process
struct FS{T} <: ContinuousTimeProcess{ℝ{2,T}}
    α::T
    β::T
    λ::T
    μ::T
    σ1::T
    σ2::T
end

dose(t) = 2*(t/2)/(1+(t/2)^2)

b(t, x, P::FS) = ℝ{2}(P.α*dose(t) -(P.λ + P.β)*x[1] + P.μ*x[2],  P.λ*x[1] -P.μ*x[2])  # reminder mu = k-lambda
σ(t, x, P::FS) = @SMatrix [P.σ1 0.0 ;0.0  P.σ1]
constdiff(::FS) = true
clone(::FS, θ) = FS(θ...)
params(P::FS) = [P.α, P.β, P.λ, P.μ, P.σ1, P.σ2]


################## specify auxiliary process
struct FSAux{T,S} <: ContinuousTimeProcess{ℝ{2, T}}
    α::T
    β::T
    λ::T
    μ::T
    σ1::T
    σ2::T
    t::T
    u::S
    T::T
    v::S
end



B(t, P::FSAux) = @SMatrix [ -P.λ - P.β  P.μ ;  P.λ  -P.μ]
#B(t, P::FSAux) = @SMatrix [ 0.0  0.0 ;  0.0 0.0]
β(t, P::FSAux) = ℝ{2}(P.α*dose(t),0.0)
σ(t, P::FSAux) = @SMatrix [P.σ1 0.0 ;0.0   P.σ1]
constdiff(::FSAux) = true
b(t, x, P::FSAux) = B(t,P) * x + β(t,P)
a(t, P::FSAux) = σ(t, P) * σ(t, P)'
clone(P::FSAux, θ) = FSAux(θ..., P.t, P.u, P.T, P.v)
