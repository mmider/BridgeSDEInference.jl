mkpath("output/")
outdir="output"

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using DataFrames
using CSV
#const ℝ = SVector
# specify observation scheme
L = @SMatrix [1. 0.;
              0. 1.]
Σ = @SMatrix [10^(-5) 0.;
     0. 10^(-5)]

# choose parametrisation of the FitzHugh-Nagumo
POSSIBLE_PARAMS = [:regular, :simpleAlter, :complexAlter, :simpleConjug,
                   :complexConjug]
parametrisation = POSSIBLE_PARAMS[5]
include("src/fitzHughNagumo.jl")

#NOTE important! MCMCBridge must be imported after FHN is loaded
#include("src/MCMCBridge.jl")
#using Main.MCMCBridge


include("src/types.jl")
include("src/ralston3.jl")
include("src/rk4.jl")
include("src/tsit5.jl")
include("src/vern7.jl")

include("src/guid_prop_bridge.jl")
include("src/blocking_schedule.jl")

# Initial parameter guess.
#θ₀ = [0.1, 0.0, 1.5, 0.8, 0.3]
θ₀ = [10.0, -8.0, 15.0, 0.0, 3.0]

# Target law
P˟ = FitzhughDiffusion(θ₀...)
# Auxiliary law
t₀ = 1.0
T = 2.0
x0 = ℝ{2}(-0.5, 2.25)
xT = ℝ{2}(1.0, 0.0)

P̃ = FitzhughDiffusionAux(θ₀..., t₀, L*x0, T, L*xT)

τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
dt = 1/10000
tt = τ(t₀,T).(t₀:dt:T)


changePt = NoChangePt()

P = GuidPropBridge(eltype(x0), tt, P˟, P̃, L, L*x0, Σ;
                   changePt=changePt, solver=Vern7())

P₂ = GuidPropBridge(eltype(x0), tt, P˟, P̃, L, L*x0, Σ;
                    changePt=SimpleChangePt(1000), solver=Vern7())

Hs = [h1-h2 for (h1, h2) in zip(P.H,P₂.H)]
