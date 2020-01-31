const ReadjustT = NamedTuple{(:step, :scale, :min, :max, :trgt, :offset),
                             Tuple{Int64, Float64, Float64, Float64, Float64, Int64}}

function named_readjust(p)
    @assert length(p) == 6
    (step=p[1], scale=p[2], min=p[3], max=p[4], trgt=p[5], offset=p[6])
end

"""
δ decreases roughly proportional to scale/sqrt(iteration)
"""
compute_δ(p, mcmc_iter) = p.scale/sqrt(max(1.0, mcmc_iter/p.step-p.offset))

"""
ϵ is moved by δ to adapt to target acceptance rate
"""
function compute_ϵ(ϵ_old, p, a_r, δ, flip=1.0, f=identity, finv=identity)
    ϵ = finv(f(ϵ_old) + flip*(2*(a_r > p.trgt)-1)*δ)
    ϵ = max(min(ϵ,  p.max), p.min)    # trim excessive updates
end

sigmoid(x, a=1.0) = 1.0 / (1.0 + exp(-a*x))
logit(x, a=1.0) = (log(x) - log(1-x))/a
