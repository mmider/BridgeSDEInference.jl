using Distributions
import Random: rand!, rand
import Distributions: logpdf
import Base: eltype, length
using GaussianDistributions


#===============================================================================
                        One dimensional random walk
===============================================================================#
abstract type RandomWalk end

mutable struct UniformRandomWalk{T} <: RandomWalk
    ϵ::T
    pos::Bool

    UniformRandomWalk(ϵ::T, pos=false) where T <: Number = new{T}(ϵ, pos)
end

eltype(::UniformRandomWalk{T}) where T = T
length(::UniformRandomWalk) = 1

function _rand(rw::UniformRandomWalk, θ, coord_idx)
    @assert truelength(coord_idx) == 1
    ϑ = thetainc(coord_idx, θ)[1]
    δ = rand(Uniform(-rw.ϵ, rw.ϵ))
    ϑ += rw.pos ? 0.0 : δ
    ϑ *= rw.pos ? exp(δ) : 1.0
    ϑ
end

function rand(rw::UniformRandomWalk, θ, coord_idx)
    ϑ = _rand(rw, θ, coord_idx)
    move_to_proper_place(ϑ, θ, coord_idx)
end

function rand!(rw::UniformRandomWalk, θ, coord_idx)
    ϑ = _rand(rw, θ, coord_idx)
    move_to_proper_place!(ϑ, θ, coord_idx)
end


function logpdf(rw::UniformRandomWalk, coord_idx, θ, θᵒ)
    @assert truelength(coord_idx) == 1
    !rw.pos && return 0.0
    ϑᵒ = thetainc(coord_idx, θᵒ)[1]
    -log(2.0*rw.ϵ)-log(ϑᵒ)
end

function readjust!(rw::UniformRandomWalk, accpt_track, param, ::Any, mcmc_iter, ::Any)
    δ = compute_δ(param, mcmc_iter)
    a_r = acceptance_rate(accpt_track)
    ϵ = compute_ϵ(rw.ϵ, param, a_r, δ)
    print("Updating random walker...\n")
    print("acceptance rate: ", round(a_r, digits=2),
          ", previous ϵ: ", round(rw.ϵ, digits=3),
          ", new ϵ: ", round(ϵ, digits=3), "\n")
    rw.ϵ = ϵ
    ϵ
end

get_dispersion(rw::UniformRandomWalk) = copy(rw.ϵ)
#===============================================================================
                    Multidimensional Gaussian random walk
===============================================================================#
mutable struct GaussianRandomWalk{T} <: RandomWalk
    Σ::Array{T,2}
    pos::Vector{Bool}

    function GaussianRandomWalk(Σ::Array{T,2}, pos=nothing) where T
        @assert size(Σ)[1] == size(Σ)[2]
        pos = (pos === nothing) ? fill(false, size(Σ)[1]) : pos
        new{T}(Σ, pos)
    end
end

eltype(::GaussianRandomWalk{T}) where T = T
length(rw::GaussianRandomWalk) = length(rw.pos)
remove_constraints!(rw::GaussianRandomWalk, ϑ) = (ϑ[rw.pos] = log.(ϑ[rw.pos]))
reimpose_constraints!(rw::GaussianRandomWalk, ϑ) = (ϑ[rw.pos] = exp.(ϑ[rw.pos]))

function _rand(rw::GaussianRandomWalk, θ, coord_idx)
    ϑ = [thetainc(coord_idx, θ)...]
    remove_constraints!(rw, ϑ)
    ϑᵒ = rand(Gaussian(ϑ, rw.Σ))
    reimpose_constraints!(rw, ϑᵒ)
    ϑᵒ
end

function rand(rw::GaussianRandomWalk, θ, coord_idx)
    ϑ = _rand(rw, θ, coord_idx)
    θᵒ = copy(θ)
    move_to_proper_place!(ϑ, θᵒ, coord_idx)
end

function rand!(rw::GaussianRandomWalk, θ, coord_idx)
    ϑ = _rand(rw, θ, coord_idx)
    move_to_proper_place!(ϑ, θ, coord_idx)
end

_logjacobian(rw::GaussianRandomWalk, ϑ, coord_idx) = -sum(log.(ϑ[rw.pos]))

function logpdf(rw::GaussianRandomWalk, coord_idx, θ, θᵒ)
    ϑ = [thetainc(coord_idx, θ)...]
    ϑᵒ = [thetainc(coord_idx, θᵒ)...]
    logJ = _logjacobian(rw, ϑᵒ, coord_idx)
    remove_constraints!(rw, ϑ)
    remove_constraints!(rw, ϑᵒ)
    logpdf(Gaussian(ϑ, rw.Σ), ϑᵒ) + logJ
end

function readjust!(rw::GaussianRandomWalk, ::Any, ::Any, corr, ::Any, coord_idx)
    ρ = matinc(corr, coord_idx)
    Σ = 2.38^2/length(rw)*ρ
    print("Updating multivariate random walker...\n")
    print("correlation: ", round.(ρ, digits=2),
          ", previous Σ: ", round.(rw.Σ, digits=3),
          ", new ϵ: ", round.(Σ, digits=3), "\n")
    rw.Σ = Σ
end

#===============================================================================
            Multidimensional mixture of Gaussian random walks
===============================================================================#

struct GaussianRandomWalkMix{T} <: RandomWalk
    gsn_A::GaussianRandomWalk{T}
    gsn_B::GaussianRandomWalk{T}
    pos::Vector{Bool}
    λ::Float64

    function GaussianRandomWalkMix(Σ_A::Array{T,2}, Σ_B::Array{T,2}, λ=0.5,
                                   pos=nothing) where T
        @assert 0.0 <= λ <= 1.0
        gsn_A = GaussianRandomWalk(Σ_A, pos)
        gsn_B = GaussianRandomWalk(Σ_B, pos)
        new{T}(gsn_A, gsn_B, pos, λ)
    end
end

eltype(::GaussianRandomWalkMix{T}) where T = T
length(rw::GaussianRandomWalkMix) = length(rw.pos)

function rand(rw::GaussianRandomWalkMix, θ, coord_idx)
    rw_i = pick_kernel(rw)
    rand(rw_i, θ, coord_idx)
end

function rand!(rw::GaussianRandomWalkMix, θ, coord_idx)
    rw_i = pick_kernel(rw)
    rand!(rw_i, θ, coord_idx)
end

function logpdf(rw::GaussianRandomWalk, coord_idx, θ, θᵒ)
    log( (1-rw.λ)*exp(logpdf(rw.gsn_A, coord_idx, θ, θᵒ))
          + rw.λ *exp(logpdf(rw.gsn_B, coord_idx, θ, θᵒ)) )
end

function readjust!(rw::GaussianRandomWalkMix, ::Any, ::Any, cov_mat, ::Any, coord_idx)
    readjust!(rw.gsn_A, nothing, nothing, cov_mat, nothing, coord_idx)
end
