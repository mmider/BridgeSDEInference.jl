module MCMCBridge

using ForwardDiff
using ForwardDiff: value


# types.jl
export ImproperPrior, idx

# mcmc.jl
export mcmc, PartObs, FPT, FPTInfo, ConjugateUpdt, MetropolisHastingsUpdt

# ODE solvers:
export Ralston3, RK4, Tsit5, Vern7

# random_walk.jl
export RandomWalk

# save_to_files.jl
export savePathsToFile, saveChainToFile

# priors.jl
export Priors


include("types.jl")
include("ralston3.jl")
include("rk4.jl")
include("tsit5.jl")
include("vern7.jl")

include("priors.jl")

include("guid_prop_bridge.jl")
include("random_walk.jl")
include("mcmc.jl")

include("save_to_files.jl")

end
