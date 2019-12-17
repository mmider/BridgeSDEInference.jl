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
include(joinpath(SRC_DIR, "radial_ornstein_uhlenbeck.jl"))
include(joinpath(SRC_DIR, "euler_maruyama_dom_restr.jl"))
include(joinpath(SRC_DIR, "prokaryotic_autoregulatory_gene_network.jl"))

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
# values taken from table 1 of Golithly and Wilkinson
θ₀ = [0.1, 0.7, 0.35, 0.2, 0.1, 0.9, 0.3, 0.1]
K = 10.0
Pˣ = Prokaryote(θ₀..., K)

x0, dt, T = ℝ{4}(7.0, 10.0, 4.0, 6.0), 1/5000, 50.0
tt = 0.0:dt:T
XX, _ = simulate_segment(ℝ{4}(0.0, 0.0, 0.0, 0.0), x0, Pˣ, tt)

using Plots

x₁ = [x[1] for x in XX.yy]
x₂ = [x[2] for x in XX.yy]
x₃ = [x[3] for x in XX.yy]
x₄ = [x[4] for x in XX.yy]

skip = 10
plot(XX.tt[1:skip:end], x₁[1:skip:end])
plot(XX.tt[1:skip:end], x₂[1:skip:end])
plot(XX.tt[1:skip:end], x₃[1:skip:end])
plot(XX.tt[1:skip:end], x₄[1:skip:end])


# Let's try proposal now
idx₁, idx₂ = 100001, 105001
t0, T = XX.tt[idx₁], XX.tt[idx₂]
x0, xT = XX.yy[idx₁], XX.yy[idx₂]


P̃ = ProkaryoteAux(θ₀..., K, t0, x0, T, xT, Val{(true, true, true, true)}())
tt = t0:dt:T


XX2, _ = simulate_segment(ℝ{4}(0.0, 0.0, 0.0, 0.0), x0, P̃, tt)

[x[1] for x in XX2.yy]

plot(XX.tt[idx₁:idx₂], x₁[idx₁:idx₂])
plot!(XX2.tt, [x[1] for x in XX2.yy])
plot(XX.tt[idx₁:idx₂], x₂[idx₁:idx₂])
plot!(XX2.tt, [x[2] for x in XX2.yy])
plot(XX.tt[idx₁:idx₂], x₃[idx₁:idx₂])
plot!(XX2.tt, [x[3] for x in XX2.yy])
plot(XX.tt[idx₁:idx₂], x₄[idx₁:idx₂])
plot!(XX2.tt, [x[4] for x in XX2.yy])

τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
tt = τ(t0,T).(tt)
Σdiagel = 10^-4
Σ = SMatrix{4,4}(1I)*Σdiagel
L = SMatrix{4,4}(1I)


Pᵒ = GuidPropBridge(eltype(x0), tt, Pˣ, P̃, L, xT, Σ; solver=Vern7())

XX3, _ = simulate_segment(ℝ{4}(0.0, 0.0, 0.0, 0.0), x0, Pᵒ, tt)

for i in 1:1000
    XX3, _ = simulate_segment(ℝ{4}(0.0, 0.0, 0.0, 0.0), x0, Pᵒ, tt)
end


plot(XX.tt[idx₁:idx₂], x₁[idx₁:idx₂])
plot!(XX3.tt, [x[1] for x in XX3.yy])
plot(XX.tt[idx₁:idx₂], x₂[idx₁:idx₂])
plot!(XX3.tt, [x[2] for x in XX3.yy])
plot(XX.tt[idx₁:idx₂], x₃[idx₁:idx₂])
plot!(XX3.tt, [x[3] for x in XX3.yy])
plot(XX.tt[idx₁:idx₂], x₄[idx₁:idx₂])
plot!(XX3.tt, [x[4] for x in XX3.yy])
