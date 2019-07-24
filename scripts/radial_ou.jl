include("../src/auxiliary/data_simulation_fns.jl")
include("../src/bounded_diffusion_domain.jl")
include("../src/radial_ornstein_uhlenbeck.jl")
include("../src/euler_maruyama.jl")
using Random
using Plots

θ = [2.0, √2.0]
Pˣ = RadialOU(θ...)

dt = 1/5000
T = 1.0
tt = 0.0:dt:T

x0 = ℝ{1}(0.5)

#Random.seed!(4)
for i in 1:100
    start = time()
    XX, _ = simulateSegment(0.0, x0, Pˣ, tt)
    elapsed = time() - start
    print("elapsed: ", elapsed, "\n")
end

num_obs = 1000
skip = div(length(tt), num_obs)
Time = collect(tt)[1:skip:end]
path = [x[1] for x in XX.yy[1:skip:end]]


plot(Time, path)
