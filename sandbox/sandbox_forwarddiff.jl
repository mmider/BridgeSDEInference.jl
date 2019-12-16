#NOTE FILE TO BE DELETED

SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR = joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
include(joinpath(SRC_DIR, "fitzHughNagumo.jl"))
include(joinpath(SRC_DIR, "conjugateUpdt.jl"))

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
Σdiagel = 10^(-4)
Σ = @SMatrix [Σdiagel]

using ForwardDiff
import ForwardDiff: Dual, Tag
tt = 0.0:0.01:2.0
Wnr = Wiener{Float64}()
W = Bridge.samplepath(tt, zero(Float64))
sample!(W, Wnr)
y = ℝ(0.0, 0.0)
CT = Tag{Val{:custom_tag}, Float64}
y_dual = ℝ(Dual{CT}(0.0,0.0), Dual{CT}(0.0,0.0))
X = SamplePath(tt, zeros(typeof(y_dual), length(tt)))
# compute log likelihood for a given wiener path and parameter vector
function ll(θ)
    for t in θ
        print(t, "\n\n")
    end
    P˟ = FitzhughDiffusion(param, θ...)
    P̃ = FitzhughDiffusionAux(param, θ..., 0.0, ℝ(Dual{CT}(0.0,0.0)),
                                          2.0, ℝ(Dual{CT}(0.5,0.0)))
    L = @SMatrix [Dual{CT}(1.0,0.0) Dual{CT}(0.0,0.0)]
    Σ = @SMatrix [Dual{CT}(10^(-4),0.0)]
    P = GuidPropBridge(Dual{CT,Float64,1}, tt, P˟, P̃, L, ℝ(Dual{CT}(0.5,0.0)), Σ;
                       changePt=NoChangePt(), solver=Vern7())
    solve!(EulerMaruyamaBounded(), X, y_dual, W, P)
    loglik = llikelihood(LeftRule(), X, P)
    loglik
end

chunkSize = 1
result = DiffResults.GradientResult(θ₀)
cfg = ForwardDiff.GradientConfig(ll, θ₀, ForwardDiff.Chunk{chunkSize}(), CT())
@time ForwardDiff.gradient!(result, ll, θ₀, cfg, Val{false}()) # turn off tag checking..., this needs to be solved somehow in the future

result.value
result.derivs


Dual{CT}.(L, 0.0)


using Interpolations
using StaticArrays
itp = LinearInterpolation([0.0, 1.0, 3.0], [ℝ(1.0, 2.0), ℝ(3.0,-2.0), ℝ(2.0,0.0)])

itp = LinearInterpolation([0.0, 1.0], [ℝ(0.0), ℝ(0.0)], extrapolation_bc = Line())
itp(2.0)

using Plots
tt = collect(0.0:0.01:3.0)
xx₁ = [x[1] for x in itp(tt)]
xx₂ = [x[2] for x in itp(tt)]
plot(tt, xx₁)
plot!(tt, xx₂)



LinearInterpolation([0.0, 1.0, 3.0], [ℝ(1.0, 2.0), ℝ(3.0,-2.0), ℝ(2.0,0.0)])
