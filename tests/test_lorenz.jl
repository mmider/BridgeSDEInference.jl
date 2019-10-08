SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
#include(joinpath(SRC_DIR, "fitzHughNagumo.jl"))
#include(joinpath(SRC_DIR, "fitzHughNagumo_conjugateUpdt.jl"))

include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "vern7.jl"))
include(joinpath(SRC_DIR, "tsit5.jl"))
include(joinpath(SRC_DIR, "rk4.jl"))
include(joinpath(SRC_DIR, "ralston3.jl"))
include(joinpath(SRC_DIR, "priors.jl"))
include(joinpath(SRC_DIR, "guid_prop_bridge.jl"))

include(joinpath(SRC_DIR, "bounded_diffusion_domain.jl"))
include(joinpath(SRC_DIR, "lorenz_system.jl"))

include(joinpath(SRC_DIR, "random_walk.jl"))
include(joinpath(SRC_DIR, "blocking_schedule.jl"))
include(joinpath(SRC_DIR, "starting_pt.jl"))
include(joinpath(SRC_DIR, "mcmc.jl"))
include(joinpath(SRC_DIR, "path_to_wiener.jl"))

using StaticArrays
using Distributions # to define priors
using Random        # to seed the random number generator

# Let's generate the data
# -----------------------
using Bridge
include(joinpath(AUX_DIR, "data_simulation_fns.jl"))
Random.seed!(4)
θ₀ = [10.0, 28.0, 8.0/3.0, 3.0, 3.0, 3.0]
Pˣ = Lorenz(θ₀...)

x0, dt, T = ℝ{3}(1.5, -1.5, 25.0), 1/5000, 4.0
tt = 0.0:dt:T
XX, _ = simulateSegment(ℝ{3}(0.0, 0.0, 0.0), x0, Pˣ, tt)
XX
using Plots

plt = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),
             title="Lorenz Attractor", marker = 2)
idxes = [i for i in 1:100:length(tt)]
@gif for i in idxes
    push!(plt, XX.yy[i][1], XX.yy[i][2], XX.yy[i][3])
end every 1
