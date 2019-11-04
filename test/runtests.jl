using Test, Suppressor
using Bridge, BridgeSDEInference, StaticArrays, Distributions
using Statistics, Random, LinearAlgebra

const BSI = BridgeSDEInference
using BridgeSDEInference: ℝ

include("test_ODE_solver_change_pt.jl")
include("test_blocking.jl")
include("test_measchange.jl")
include("test_random_walk.jl")
include("test_mcmc_components.jl")
include("test_workspace.jl")
include("test_setup.jl")
