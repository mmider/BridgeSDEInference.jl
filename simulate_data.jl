mkpath("output/")
outdir="output/"
using Bridge
using DataFrames
using CSV


"""
    simulateSegment(::S, x0, P, tt)

Simulate path of a target process `P`, started from `x0` on a time-grid `tt`
"""
function simulateSegment(::S, x0, P, tt) where S
    Wnr = Wiener{S}()
    WW = Bridge.samplepath(tt, zero(S))
    sample!(WW, Wnr)
    XX = solve(Euler(), x0, WW, P)
    XX, XX.yy[end]
end

"""
    findCrossings(XX, upLvl, downLvl, upSearch=false)

Find times of first up-crossings of level `upLvl`. The timing starts only after
the diffusion was brought below level `downLvl` first (or alternatively if
`upSearch` flag is set to true)
"""
function findCrossings(XX, upLvl, downLvl, upSearch=false)
    upSearch=upSearch
    upCrossingTimes = Float64[0.0]
    for (x,t) in zip(XX.yy, XX.tt)
        if upSearch && x[1] > upLvl
            upSearch = false
            push!(upCrossingTimes, t)
        elseif !upSearch && x[1] < downLvl
            upSearch = true
        end
    end
    upCrossingTimes, upSearch
end

# Set up process

POSSIBLE_PARAMS = [:regular, :simpleAlter, :simpleConjug]
parametrisation = POSSIBLE_PARAMS[3]
include("src/fitzHughNagumo.jl")
#P = FitzhughDiffusion(0.1, 0.0, 1.5, 0.8, 0.3)
#P = FitzhughDiffusion(0.4, 0.05, 1.2, 0.3, 0.2)
P = FitzhughDiffusion(10.0, -8.0, 25.0, 0.0, 3.0)
x0 = ℝ{2}(-0.5, 0.6)
if parametrisation == :simpleAlter
    x0 = regularToAlter(x0, P.ϵ, 0.0)
elseif parametrisation == :simpleConjug
    x0 = regularToConjug(x0, P.ϵ, 0.0)
end
L = @SMatrix [1. 0.]

dt = 1/50000
T = 10.0
tt = 0.0:dt:T

# ------------------------------ #
#  Partially observed diffusion  #
# ------------------------------ #
XX, _ = simulateSegment(0.0, x0, P, tt)
XX.yy[1]
skip = 50000
Time = collect(tt)[1:skip:end]
x1 = [(L*x)[1] for x in XX.yy[1:skip:end]]
x2 = [NaN for t in Time]
x2[1] = x0[2]
df = DataFrame(time=Time, x1=x1, x2=x2)
CSV.write(outdir*"sparse_path_part_obs_"*String(parametrisation)*".csv", df)

# --------------------------------- #
#  First passage time observations  #
# --------------------------------- #
# NOTE DEPRECATED
recentlyUpSearch = true
df_all = DataFrame(time=[], upCross=[], downCross=[], x2=[])
tt_temp = copy(tt)

XX, _ = simulateSegment(0.0, x0, P, tt_temp)
upLvl = 0.5
downLvl = -0.5
upCrossingTimes, recentlyUpSearch = findCrossings(XX, upLvl, downLvl, recentlyUpSearch)
upCross = collect(Iterators.flatten([[x0[1]],
                                     fill(upLvl, length(upCrossingTimes)-1)]))
downCross = fill(downLvl, length(upCrossingTimes))
x2 = fill(NaN, length(upCrossingTimes))
x2[1] = x0[2]
XX
df = DataFrame(time=upCrossingTimes,
               upCross=upCross,
               downCross=downCross,
               x2=x2)
df_all = vcat(df_all, df)
x0 = XX.yy[end]
tt_temp = tt .+ tt_temp[end]

df_all
CSV.write(outdir*"up_crossing_times_exmple_"*String(parametrisation)*".csv", df_all)
