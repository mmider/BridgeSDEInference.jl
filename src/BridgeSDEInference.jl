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

# setup.jl
export MCMCSetup, set_observations!, set_imputation_grid!, set_transition_kernels!
export set_priors!, set_mcmc_params!, set_blocking!, set_solver!, initialise!

export Workspace

include("types.jl")

_DIR = "stochastic_process"
include(joinpath(_DIR, "bounded_diffusion_domain.jl"))
include(joinpath(_DIR, "guid_prop_bridge.jl"))
include(joinpath(_DIR, "path_to_wiener.jl"))

_DIR = "solvers"
include(joinpath(_DIR, "vern7.jl"))
include(joinpath(_DIR, "tsit5.jl"))
include(joinpath(_DIR, "rk4.jl"))
include(joinpath(_DIR, "ralston3.jl"))
include(joinpath(_DIR, "euler_maruyama_dom_restr.jl"))

_DIR = "transition_kernels"
include(joinpath(_DIR, "random_walk.jl"))

_DIR = "mcmc_extras"
include(joinpath(_DIR, "adaptation.jl"))
include(joinpath(_DIR, "blocking_schedule.jl"))
include(joinpath(_DIR, "first_passage_times.jl"))
include(joinpath(_DIR, "starting_pt.jl"))

_DIR = "examples"
include(joinpath(_DIR, "fitzhugh_nagumo.jl"))
include(joinpath(_DIR, "radial_ornstein_uhlenbeck.jl"))
include(joinpath(_DIR, "lorenz_system.jl"))
include(joinpath(_DIR, "lorenz_system_const_vola.jl"))

_DIR = "mcmc"
include(joinpath(_DIR, "priors.jl"))
include(joinpath(_DIR, "setup.jl"))
include(joinpath(_DIR, "workspace.jl"))
include(joinpath(_DIR, "conjugate_updt.jl"))
include(joinpath(_DIR, "mcmc.jl"))
include(joinpath(_DIR, "repeated.jl"))

end
