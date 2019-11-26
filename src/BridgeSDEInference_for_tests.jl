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
include(joinpath(_DIR, "fusion.jl"))
include(joinpath(_DIR, "conjugate_updt.jl"))
include(joinpath(_DIR, "updates.jl"))
include(joinpath(_DIR, "mcmc.jl"))
#include(joinpath(_DIR, "repeated.jl"))
