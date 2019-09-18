include("../src/auxiliary/data_simulation_fns.jl")
include("../src/bounded_diffusion_domain.jl")
include("../src/radial_ornstein_uhlenbeck.jl")
include("../src/euler_maruyama_dom_restr.jl")
using Random
using Plots

θ = [0.05, √2.0]
Pˣ = RadialOU(θ...)

dt = 1/5000
T = 1.0
tt = 0.0:dt:T

x0 = ℝ{1}(0.5)


N = 100
samples = Vector{Any}(undef, N)
Random.seed!(4)
for i in 1:N
    XX, _ = simulateSegment(0.0, x0, Pˣ, tt)
    samples[i] = XX
end

num_obs = 1000
skip = div(length(tt), num_obs)
Time = collect(tt)[1:skip:end]
path = [x[1] for x in samples[1].yy[1:skip:end]]


p = plot(Time, path, alpha=0.3, color="steelblue", label="")
for i in 2:N
    path = [x[1] for x in samples[i].yy[1:skip:end]]
    plot!(Time, path, alpha=0.3, color="steelblue", label="")
end
p
