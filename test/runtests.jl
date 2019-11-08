using Test, Suppressor
using Bridge, StaticArrays, Distributions
using Statistics, Random, LinearAlgebra

#SRC_DIR = joinpath(Base.source_dir(), "..", "src")
#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
#BSI = Main.BridgeSDEInference

using BridgeSDEInference
const BSI = BridgeSDEInference
using BridgeSDEInference: ℝ

include("test_ODE_solver_change_pt.jl")
include("test_blocking.jl")
include("test_measchange.jl")
include("test_random_walk.jl")
include("test_mcmc_components.jl")
include("test_workspace.jl")
include("test_setup.jl")
include("test_fusion.jl")
