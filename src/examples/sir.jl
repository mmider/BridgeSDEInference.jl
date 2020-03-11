using Bridge
using StaticArrays
import Bridge: b, σ, B, β, a, constdiff
const ℝ = SVector{N,T} where {N,T}
import Base.display
sq(x) = sqrt(max(x, 2e-10))
struct SIR{T} <: ContinuousTimeProcess{SVector{2,T}}
    α::T
    β::T
    σ1::T
    σ2::T
end

b(t, u, P::SIR) = @SVector [P.α*(1 - u[1] - u[2])*u[1] - P.β*u[1], P.β*u[1]]
σ(t, u, P::SIR) = @SMatrix Float64[
    -P.σ1*sq((1 - u[1] - u[2])*u[1])  -P.σ2*sq(u[1])
                0.0   P.σ2*sq.(u[1])
    ]
a(t, u, P::SIR) = σ(t, u, P)*σ(t, u, P)'
constdiff(::SIR) = false
clone(P::SIR, θ) = SIR(θ...)
params(P::SIR) = [P.α, P.β, P.σ1, P.σ2]
#domain(P::SIR) = LowerBoundedDomain((0.0, 0.0), (1,2))
domain(P::SIR) = LowerBoundedDomain((0.0,), (1,))


# <---------------------------------------------
# this is optional, needed for conjugate updates
phi(::Val{0}, t, u, P::SIR) = (zero(u[1]), zero(u[2]))
#[P.α*(P.k - u[1] - u[2])*u[1] - P.β*u[1], P.β*u[1]]
phi(::Val{1}, t, u, P::SIR) = ((P.k - u[1] - u[2])*u[1], zero(u[2]))
phi(::Val{2}, t, u, P::SIR) = (-u[1], u[1])
phi(::Val{3}, t, x, P::SIR) = (zero(x[1]), zero(x[2]))
phi(::Val{4}, t, x, P::SIR) = (zero(x[1]), zero(x[2]))
phi(::Val{5}, t, x, P::SIR) = (zero(x[1]), zero(x[2]))



nonhypo(P::SIR, x) = x
@inline hypo_a_inv(P::SIR, t, x) = inv(a(t, x, P))
num_non_hypo(P::Type{<:SIR}) = 2


struct SIRAux{T,S1,S2} <: ContinuousTimeProcess{SVector{2,T}}
    α::T
    β::T
    σ1::T
    σ2::T
    t::Float64
    u::S1
    T::Float64
    v::S2
end


# function B(t, P::SIRAux)
# #    b(t, u, P::SIR) = @SVector [P.α*(P.k - u[1] - u[2])*u[1] - P.β*u[1], P.β*u[1]]
#     @SMatrix [(P.α) -(P.β);
#               (P.β) -0.0]
# end
function B(t, P::SIRAux)
#    b(t, u, P::SIR) = @SVector [P.α*(1 - u[1] - u[2])*u[1] - P.β*u[1], P.β*u[1]]
    @SMatrix [(P.α*(1 - P.v[1] - P.v[2]) - (P.β))  0.0;
              (P.β) 0.0]
end





# mean = ℝ{2}(P.γ/P.δ, P.α/P.β)
function β(t, P::SIRAux)
    ℝ{2}(0.0, 0.0)
end

# function σ(t, P::SIRAux)
#     sq(0.0001)*@SMatrix Float64[
#          -P.σ1  -P.σ2
#           0.0   P.σ2
#         ]
# end

function σ(t, P::SIRAux)
        @SMatrix Float64[
         (-P.σ1*(1 - P.v[2] - P.v[1])*P.v[1])  -P.σ2*P.v[1];
         0.0   P.σ2*P.v[1]]
end




σ(t, x, P::SIRAux) = σ(t, P)

depends_on_params(::SIRAux) = (3, 4, 5)

constdiff(::SIRAux) = true
b(t, x, P::SIRAux) = B(t,P) * x + β(t,P)
a(t, P::SIRAux) = σ(t,P) * σ(t, P)'


clone(P::SIRAux, θ) = SIRAux(θ..., P.t, P.u, P.T, P.v)

clone(P::SIRAux, θ, v) = SIRAux(θ..., P.t, v, P.T, v)
params(P::SIRAux) = [P.α, P.β, P.σ1, P.σ2]
