using Distributions
import Random: rand!, rand
import Distributions: logpdf

struct MALA{T}
    ϵ::T
    MALA(ϵ::T) where T = new{T}(ϵ)
end

function rand!(mala::MALA, θ, ∇ll, ::UpdtIdx) where UpdtIdx
    id = [idx(UpdtIdx())...]
    θ[id] .+= mala.ϵ .* ∇ll .+ sqrt.(2.0.*mala.ϵ).*randn(length(∇ll))
    θ
end

function rand(mala::MALA, θ, ∇ll, ::UpdtIdx) where UpdtIdx
    θc = copy(θ)
    rand!(mala, θc, ∇ll, UpdtIdx())
end

function logpdf(mala::MALA, θ, θᵒ, ∇ll, ::UpdtIdx) where UpdtIdx
    id = [idx(UpdtIdx())...]
    logpdf(MvNormal(θ[id] .+ mala.ϵ.*∇ll, diagm(0=>sqrt.(2.0.*mala.ϵ))), θᵒ[id])
end
