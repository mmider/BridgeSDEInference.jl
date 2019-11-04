module BridgeSDEInference

using Bridge, StaticArrays, Distributions
using Statistics, Random, LinearAlgebra
using ForwardDiff
using ForwardDiff: value

# fitzHughNagumo.jl
export FitzhughDiffusion, FitzhughDiffusionAux, ℝ
export regularToAlter, alterToRegular, regularToConjug, conjugToRegular, display

# types.jl
export ImproperPrior, ImproperPosPrior, NoChangePt, SimpleChangePt

# mcmc.jl
export mcmc, PartObs, FPT, FPTInfo, ConjugateUpdt, MetropolisHastingsUpdt

# ODE solvers:
export Ralston3, RK4, Tsit5, Vern7

# euler_maruyama_dom_restr.jl
export forcedSolve!, forcedSolve

# random_walk.jl
export UniformRandomWalk, GaussianRandomWalk

# priors.jl
export Priors

# blocking_schedule.jl
export NoBlocking, ChequeredBlocking, create_blocks

# starting_pt.jl
export KnownStartingPt, GsnStartingPt

# radial_ornstein_uhlenbeck.jl
export RadialOU, RadialOUAux

# lorenz_system.jl
export Lorenz, LorenzAux

# lorenz_system_const_vola.jl
export LorenzCV, LorenzCVAux

# prokaryotic_autoregulatory_gene_network.jl
export Prokaryote, ProkaryoteAux

# adaptation.jl
export Adaptation, NoAdaptation

# setup.jl
export MCMCSetup, DiffusionSetup, set_observations!, set_imputation_grid!
export set_x0_prior!, initialise!, set_auxiliary!

export MCMCSchedule

export Workspace

_DIR = "general"
include(joinpath(_DIR, "types.jl"))
include(joinpath(_DIR, "coordinate_access.jl"))
include(joinpath(_DIR, "readjustments.jl"))

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
include(joinpath(_DIR, "pseudo_conjugate.jl"))

_DIR = "mcmc_extras"
include(joinpath(_DIR, "adaptation.jl"))
include(joinpath(_DIR, "blocking.jl"))
include(joinpath(_DIR, "first_passage_times.jl"))
include(joinpath(_DIR, "starting_pt.jl"))

_DIR = "examples"
include(joinpath(_DIR, "fitzhugh_nagumo.jl"))
include(joinpath(_DIR, "radial_ornstein_uhlenbeck.jl"))
include(joinpath(_DIR, "lorenz_system.jl"))
include(joinpath(_DIR, "lorenz_system_const_vola.jl"))
include(joinpath(_DIR, "prokaryotic_autoregulatory_gene_network.jl"))

_DIR = "mcmc"
include(joinpath(_DIR, "priors.jl"))
include(joinpath(_DIR, "mcmc_components.jl"))
include(joinpath(_DIR, "setup.jl"))
include(joinpath(_DIR, "workspace.jl"))
include(joinpath(_DIR, "conjugate_updt.jl"))
include(joinpath(_DIR, "updates.jl"))
include(joinpath(_DIR, "mcmc.jl"))
#include(joinpath(_DIR, "repeated.jl"))

end
