using BridgeSDEInference
using Random
using Bridge
using PyPlot

P = FitzhughDiffusion(:regular, 0.1, -0.8, 1.5, 0.0, 0.3)
x0 = ℝ{2}(-0.5, -0.6)
tt = 0.0:(1/50000):10.0
Random.seed!(4)

XX, _ = simulate_segment(0.0, x0, P, tt)
plot(XX.tt, XX.yy)
