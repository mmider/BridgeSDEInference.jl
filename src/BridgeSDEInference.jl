module BridgeSDEInference

using Bridge, StaticArrays, Distributions
using Statistics, Random, LinearAlgebra
using ForwardDiff
using ForwardDiff: value

# fitzHughNagumo.jl
export FitzhughDiffusion, FitzhughDiffusionAux, ℝ
export regularToAlter, alterToRegular, regularToConjug, conjugToRegular, display

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

# radial_ornstein_uhlenbeck.jl
export RadialOU, RadialOUAux

# lorenz_system.jl
export Lorenz, LorenzAux

# lorenz_system_const_vola.jl
export LorenzCV, LorenzCVAux

# adaptation.jl
export Adaptation, NoAdaptation


include("types.jl")
include(joinpath("solvers", "ralston3.jl"))
include(joinpath("solvers", "rk4.jl"))
include(joinpath("solvers", "tsit5.jl"))
include(joinpath("solvers", "vern7.jl"))

include(joinpath("mcmc", "priors.jl"))
include(joinpath("stochastic_process", "guid_prop_bridge.jl"))
include(joinpath("mcmc", "conjugateUpdt.jl"))

include(joinpath("stochastic_process", "bounded_diffusion_domain.jl"))
include(joinpath("solvers", "euler_maruyama_dom_restr.jl"))

include(joinpath("examples", "radial_ornstein_uhlenbeck.jl"))
include(joinpath("examples", "lorenz_system.jl"))
include(joinpath("examples", "lorenz_system_const_vola.jl"))

include(joinpath("transition_kernels", "random_walk.jl"))
include(joinpath("mcmc_extras", "blocking_schedule.jl"))
include(joinpath("mcmc_extras", "starting_pt.jl"))
include(joinpath("mcmc_extras", "adaptation.jl"))
include(joinpath("mcmc", "mcmc.jl"))
include(joinpath("stochastic_process", "path_to_wiener.jl"))



end
