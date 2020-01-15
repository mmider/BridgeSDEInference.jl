using Bridge
using Random
using DataFrames
using CSV

SRC_DIR = joinpath(Base.source_dir(), "..","..", "src")
include(joinpath(SRC_DIR, "auxiliary", "data_simulation_fns.jl"))
OUT_DIR = joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

include(joinpath(SRC_DIR, "examples", "fitzhugh_nagumo.jl"))
param = :simpleConjug
FILENAME_OUT = joinpath(OUT_DIR,
                     "test_path_part_obs_"*String(param)*".csv")

P = FitzhughDiffusion(param, 10.0, -8.0, 15.0, 0.0, 3.0)
# starting point under :regular parametrisation
x0 = ℝ{2}(-0.5, 0.6)
# translate to conjugate parametrisation
x0 = regularToConjug(x0, P.ϵ, 0.0)

dt = 1/50000
T = 10.0
tt = 0.0:dt:T

Random.seed!(4)
XX, _ = simulate_segment(0.0, x0, P, tt)

num_obs = 100
skip = div(length(tt), num_obs)
Time = collect(tt)[1:skip:end]
df = DataFrame(time=Time, x1=[x[1] for x in XX.yy[1:skip:end]],
            x2=[(i==1 ? x0[2] : NaN) for (i,t) in enumerate(Time)])
CSV.write(FILENAME_OUT, df)
