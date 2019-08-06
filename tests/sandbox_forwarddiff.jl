#NOTE FILE TO BE DELETED
SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR = joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
include(joinpath(SRC_DIR, "fitzHughNagumo.jl"))
include(joinpath(SRC_DIR, "fitzHughNagumo_conjugateUpdt.jl"))

include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "vern7.jl"))
include(joinpath(SRC_DIR, "tsit5.jl"))
include(joinpath(SRC_DIR, "rk4.jl"))
include(joinpath(SRC_DIR, "ralston3.jl"))
include(joinpath(SRC_DIR, "priors.jl"))
include(joinpath(SRC_DIR, "guid_prop_bridge.jl"))

include(joinpath(SRC_DIR, "bounded_diffusion_domain.jl"))
include(joinpath(SRC_DIR, "euler_maruyama_dom_restr.jl"))
include(joinpath(SRC_DIR, "lorenz_system.jl"))
include(joinpath(SRC_DIR, "lorenz_system_const_vola.jl"))

include(joinpath(SRC_DIR, "random_walk.jl"))
include(joinpath(SRC_DIR, "blocking_schedule.jl"))
include(joinpath(SRC_DIR, "starting_pt.jl"))
include(joinpath(SRC_DIR, "mcmc.jl"))
include(joinpath(SRC_DIR, "path_to_wiener.jl"))
include(joinpath(SRC_DIR, "metropolis_adjusted_langevin_kernel.jl"))


using Distributions # to define priors
using Random        # to seed the random number generator

param = :complexConjug
# Initial parameter guess.
θ₀ = [10.0, -8.0, 15.0, 0.0, 3.0]
# Target law
P˟ = FitzhughDiffusion(param, θ₀...)
# Auxiliary law
P̃ = FitzhughDiffusionAux(param, θ₀..., 0.0, ℝ(0.0), 2.0, ℝ(0.5))

L = @SMatrix [1. 0.]
Σdiagel = 10^(-10)
Σ = @SMatrix [Σdiagel]


function ll(θ)
    P˟ = FitzhughDiffusion(param, θ...)
    P̃ = FitzhughDiffusionAux(param, θ..., 0.0, ℝ(0.0), 2.0, ℝ(0.5))
    P = GuidPropBridge(Float64, 0.0:0.01:2.0, P˟, P̃, L, ℝ(0.5), Σ;
                                     changePt=NoChangePt(), solver=Vern7())

end
Dual{ForwardDiff.Tag{typeof(ll),Float64}}(15.0,0.0)
using ForwardDiff
chunkSize = 1
result = DiffResults.GradientResult(θ₀)
cfg = ForwardDiff.GradientConfig(ll, θ₀, ForwardDiff.Chunk{chunkSize}())
ForwardDiff.gradient!(result, ll, θ₀, cfg)

result.value
result.derivs
