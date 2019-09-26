using Test

using Bridge, StaticArrays, Distributions
using Statistics, Random, LinearAlgebra
POSSIBLE_PARAMS = [:regular, :simpleAlter, :complexAlter, :simpleConjug,
                   :complexConjug]
SRC_DIR = joinpath(Base.source_dir(), "..", "src")

include("test_setup.jl")
include("test_workspace.jl")
include("test_ODE_solver_change_pt.jl")
include("test_blocking.jl")
