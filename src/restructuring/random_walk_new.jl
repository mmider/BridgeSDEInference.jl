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

    UniformRandomWalk(ϵ::T, pos=false) where T = new{T}(ϵ, pos)
end

eltype(::UniformRandomWalk{T}) where T = T
length(::UniformRandomWalk) = 1

function _rand(rw::UniformRandomWalk, θ, coord_idx)
    @assert truelength(coord_idx) == 1
    ϑ = thetainc(coord_idx, θ)[1]
    δ = rand(Uniform(-rw.ϵ, rw.ϵ))
    ϑ += rw.pos ? 0.0 : δ
    ϑ *= rw.pos ? δ : 1.0
    ϑ
end

function rand(rw::UniformRandomWalk, θ, coord_idx)
    ϑ = _rand(rw, θ, coord_idx)
    θᵒ = copy(θ)
    move_to_proper_place!(ϑ, θᵒ, coord_idx)
end

function rand!(rw::UniformRandomWalk, θ, coord_idx)
    ϑ = _rand(rw, θ, coord_idx)
    move_to_proper_place!(ϑ, θ, coord_idx)
end


function logpdf(rw::UniformRandomWalk, θ, θᵒ, coord_idx)
    @assert truelength(coord_idx) == 1
    !rw.pos && return 0.0
    ϑᵒ = thetainc(coord_idx, θᵒ)[1]
    -log(2.0*rw.ϵ)-log(ϑᵒ)
end

_compute_δ(p, mcmc_iter) = p.scale/sqrt(max(1.0, mcmc_iter/p.step-p.offset))
function _compute_ϵ(ϵ_old, p, a_r, δ, flip=1.0)
    ϵ = ϵ_old + flip*(2*(a_r > p.trgt)-1)*δ
    # trim excessive updates
    ϵ = max(min(ϵ,  p.maxϵ), p.minϵ)
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
end

#===============================================================================
                        Multidimensional random walk
===============================================================================#
mutable struct GaussianRandomWalk{T} <: RandomWalk
    Σ::Array{T,2}
    pos::Vector{Bool}

    function GaussianRandomWalk(Σ::Array{T,2}, pos=nothing)
        @assert size(Σ)[1] == size(Σ)[2]
        pos = (pos === nothing) ? fill()
        if pos === nothing
            pos = fill(false, size(Σ)[1])
        end
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
    reimpose_constraints!(rw, ϑ)
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

function logpdf(rw::GaussianRandomWalk, θ, θᵒ, coord_idx)
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
                            Fusions of random walkers
===============================================================================#
