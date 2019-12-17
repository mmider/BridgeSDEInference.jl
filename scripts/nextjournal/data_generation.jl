using BridgeSDEInference
using Random
using Bridge
using PyPlot
using StaticArrays
const State = SArray{Tuple{2},T,1,2} where T

P = FitzhughDiffusion(:regular, 0.1, -0.8, 1.5, 0.0, 0.3)
x0 = State(-0.5, -0.6)
tt = 0.0:(1/50000):50.0
Random.seed!(4)
XX, _ = simulate_segment(0.0, x0, P, tt)

num_obs = 50
skip = div(length(tt), num_obs)
obs = (time = XX.tt[1:skip:end],
       values = first.(XX.yy)[1:skip:end])
