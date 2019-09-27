using Bridge
using Random
using DataFrames
using CSV


include(joinpath("..","src","auxiliary","data_simulation_fns.jl"))
OUT_DIR = joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "fitzHughNagumo.jl"))
param = :complexConjug
FILENAME_OUT = joinpath(OUT_DIR,
                     "test_path_part_obs_"*String(param)*".csv")

K = 4

P = FitzhughDiffusion(param, 10.0, -8.0, 15.0, 0.0, 3.0)
# starting point under :regular parametrisation
# translate to conjugate parametrisation
x0 = [regularToConjug(ℝ{2}(-rand(), rand()), P.ϵ, 0.0) for k in 1:K]


dt = 1/50000
T = 10.0
tt = 0.0:dt:T
num_obs = 100
skip_ = div(length(tt), num_obs)

Random.seed!(4)
XX = [simulateSegment(0.0, x0[k], P, tt)[1][1:skip_:length(tt)-1] for k in 1:K]


#df = DataFrame(time=XX[1].tt, x1=[x[1] for x in XX[1].yy],
#            x2=[(i==1 ? x0[2] : NaN) for (i,t) in enumerate(Time)])
#CSV.write(FILENAME_OUT, df)
