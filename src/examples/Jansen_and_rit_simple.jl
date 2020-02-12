using Bridge
using StaticArrays
using LinearAlgebra
import Bridge: b, σ, B, β, a, constdiff
const ℝ = SVector{N, T} where {N, T}

"""
    JRNeuralDiffusion3n <: ContinuousTimeProcess{ℝ{6, T}}
structure defining the Jansen and Rit Neural Mass Model described in
https://mathematical-neuroscience.springeropen.com/articles/10.1186/s13408-017-0046-4 and
https://arxiv.org/abs/1903.01138
"""
struct JRNeuralDiffusion3n{T} <: ContinuousTimeProcess{ℝ{6, T}}
    A::T
    a::T
    B::T
    b::T
    C::T
    νmax::T
    v0::T
    r::T
    μx::T
    μy::T
    μz::T
    σx::T
    σy::T
    σz::T
    # constructor given assumption statistical paper
    function JRNeuralDiffusion3n(A::T, a::T, B::T, b::T, C::T,
            νmax::T, v0::T ,r::T, μx::T, μy::T, μz::T, σx::T, σy::T, σz::T) where T
        new{T}(A, a, B, b, C, νmax, v0, r, μx, μy, μz, σx, σy, σz)
    end
end
#C1 = C, C2 = 0.8C, c3 = 0.25C, c4 =  0.25C,

# in the statistical paper they set μ's to be constant and not function of time.
function μx(t, P::JRNeuralDiffusion3n)
    P.μx
end

function μy(t, P::JRNeuralDiffusion3n)
    P.μy
end

function μz(t, P::JRNeuralDiffusion3n)
    P.μz
end

"""
    sigm(x, P::JRNeuralDiffusion3n)
definition of sigmoid function
"""
function sigm(x, P::JRNeuralDiffusion3n{T}) where T
    P.νmax / (1 + exp(P.r*(P.v0 - x)))
end


function b(t, x, P::JRNeuralDiffusion3n{T}) where T
    ℝ{6, T}(x[4], x[5], x[6],
    P.A*P.a*(μx(t, P) + sigm(x[2] - x[3], P)) - 2P.a*x[4] - P.a*P.a*x[1],
    P.A*P.a*(μy(t, P) + 0.8P.C*sigm(P.C*x[1], P)) - 2P.a*x[5] - P.a*P.a*x[2],
    P.B*P.b*(μz(t, P) + 0.25P.C*sigm(0.25P.C*x[1], P)) - 2P.b*x[6] - P.b*P.b*x[3])
end


#6x3 matrix
function σ(t, x, P::JRNeuralDiffusion3n{T}) where T
    @SMatrix    [0.0  0.0  0.0;
                0.0  0.0  0.0;
                0.0 0.0  0.0;
                P.σx  0.0  0.0;
                0.0  P.σy 0.0;
                0.0  0.0  P.σz]
end

constdiff(::JRNeuralDiffusion3n) = true
clone(::JRNeuralDiffusion3n, θ) = JRNeuralDiffusion3n(θ...)
#Static vector
params(P::JRNeuralDiffusion3n) = [P.A, P.a, P.B, P.b, P.C, P.νmax,
    P.v0, P.r, P.μx, P.μy, P.μz, P.σx, P.σy, P.σz]
param_names(::JRNeuralDiffusion3n) = (:A, :a, :B, :b, :C, :νmax,
    :v0, :r, :μx, :μy, :μz, :σx, :σy, :σz)


#### Conjugate #####
#### Three dimensional ####
nonhypo(P::JRNeuralDiffusion3n, x) = x[4:6]
@inline hypo_a_inv(P::JRNeuralDiffusion3n, t, x) = SMatrix{3,3}(Diagonal([P.σx^(-2), P.σy^(-2), P.σz^(-2)]))
num_non_hypo(P::Type{<:JRNeuralDiffusion3n}) = 3

phi(::Val{0}, t, x, P::JRNeuralDiffusion3n) = (P.A*P.a*(P.μx + sigm(x[2] - x[3] , P)) - 2P.a*x[4] - P.a*P.a*x[1],
                                                P.A*P.a*0.8P.C*sigm(P.C*x[1], P) - 2P.a*x[5] - P.a*P.a*x[2],
                                                P.B*P.b*(P.μz +0.25P.C*sigm(0.25P.C*x[1], P)) - 2P.b*x[6] - P.b*P.b*x[3]
                                                )
phi(::Val{10}, t, x, P::JRNeuralDiffusion3n) = (0., P.A*P.a, 0.)
phi(::Val{1}, t, x, P::JRNeuralDiffusion3n) = (0., 0., 0.)
phi(::Val{2}, t, x, P::JRNeuralDiffusion3n) = (0., 0., 0.)
phi(::Val{3}, t, x, P::JRNeuralDiffusion3n) = (0., 0., 0.)
phi(::Val{4}, t, x, P::JRNeuralDiffusion3n) = (0., 0., 0.)
phi(::Val{5}, t, x, P::JRNeuralDiffusion3n) = (0., 0., 0.)
phi(::Val{6}, t, x, P::JRNeuralDiffusion3n) = (0., 0., 0.)
phi(::Val{7}, t, x, P::JRNeuralDiffusion3n) = (0., 0., 0.)
phi(::Val{8}, t, x, P::JRNeuralDiffusion3n) = (0., 0., 0.)
phi(::Val{9}, t, x, P::JRNeuralDiffusion3n) = (0., 0., 0.)
phi(::Val{11}, t, x, P::JRNeuralDiffusion3n) = (0., 0., 0.)
phi(::Val{12}, t, x, P::JRNeuralDiffusion3n) = (0., 0., 0.)
phi(::Val{13}, t, x, P::JRNeuralDiffusion3n) = (0., 0., 0.)
phi(::Val{14}, t, x, P::JRNeuralDiffusion3n) = (0., 0., 0.)
"""
    JRNeuralDiffusion3nAux{T, S1, S2} <: ContinuousTimeProcess{ℝ{6, T}}
structure for a simple auxiliary process defined as integrated Wiener process
"""
struct JRNeuralDiffusion3nAux{R, S1, S2} <: ContinuousTimeProcess{ℝ{6, R}}
    σx::R
    σy::R
    σz::R
    t::Float64
    u::S1
    T::Float64
    v::S2
    # generator given assumptions paper
    function JRNeuralDiffusion3nAux(σx::R, σy::R, σz::R,  t::Float64, u::S1,
                        T::Float64, v::S2) where {R, S1, S2}
        new{R, S1, S2}(σx, σy, σz, t, u, T, v)
    end
end




function B(t, P::JRNeuralDiffusion3nAux{S1, S2}) where {S1, S2}
    @SMatrix [0.0  0.0  0.0  1.0  0.0  0.0;
              0.0  0.0  0.0  0.0  1.0  0.0;
              0.0  0.0  0.0  0.0  0.0  1.0;
              0.0  0.0  0.0  0.0  0.0  0.0;
              0.0  0.0  0.0  0.0  0.0  0.0;
              0.0  0.0  0.0  0.0  0.0  0.0]
end


function β(t, P::JRNeuralDiffusion3nAux{S1, S2}) where {S1, S2}
    ℝ{6}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

#6x3 matrix
function σ(t, P::JRNeuralDiffusion3nAux{T}) where T
    @SMatrix    [0.0  0.0  0.0;
                0.0  0.0  0.0;
                0.0 0.0  0.0;
                P.σx  0.0  0.0;
                0.0  P.σy 0.0;
                0.0  0.0  P.σz]
end


b(t, x, P::JRNeuralDiffusion3nAux) = B(t,P) * x + β(t,P)
a(t, P::JRNeuralDiffusion3nAux) = σ(t,P) * σ(t, P)'

constdiff(::JRNeuralDiffusion3nAux) = true
clone(P::JRNeuralDiffusion3nAux, θ) = JRNeuralDiffusion3nAux(θ[12], θ[13], θ[14], P.t, P.u, P.T, P.v)
clone(P::JRNeuralDiffusion3nAux, θ, v) = JRNeuralDiffusion3nAux(θ[12], θ[13], θ[14], P.t, zero(v), P.T, v)
params(P::JRNeuralDiffusion3nAux) = [P.σx, P.σy, P.σz]
param_names(P::JRNeuralDiffusion3nAux) = (:σx, :σy, :σz )
depends_on_params(::JRNeuralDiffusion3nAux) = (12, 13, 14)
