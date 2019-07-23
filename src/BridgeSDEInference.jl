module BridgeSDEInference

using Bridge, StaticArrays, Distributions
using Statistics, Random, LinearAlgebra
using ForwardDiff
using ForwardDiff: value

# fitzHughNagumo.jl
export FitzhughDiffusion, FitzhughDiffusionAux
export regularToAlter, alterToRegular, regularToConjug, conjugToRegular


# types.jl
export ImproperPrior, NoChangePt, SimpleChangePt

# mcmc.jl
export mcmc, PartObs, FPT, FPTInfo, ConjugateUpdt, MetropolisHastingsUpdt

# ODE solvers:
export Ralston3, RK4, Tsit5, Vern7

# random_walk.jl
export RandomWalk

# priors.jl
export Priors

# blocking_schedule.jl
export NoBlocking, ChequeredBlocking

# starting_pt.jl
export KnownStartingPt, GsnStartingPt


include("fitzHughNagumo.jl")
include("fitzHughNagumo_conjugateUpdt.jl")

include("types.jl")
include("ralston3.jl")
include("rk4.jl")
include("tsit5.jl")
include("vern7.jl")

include("priors.jl")

include("guid_prop_bridge.jl")
include("random_walk.jl")
include("blocking_schedule.jl")
include("starting_pt.jl")
include("mcmc.jl")
include("path_to_wiener.jl")

end
