[back to README](../README.md)
# Generation of data
A short script for generating data is available [here](../simulate_data.jl). All that needs to be done is to define the law of a generating process:
```julia
POSSIBLE_PARAMS = [:regular, :simpleAlter, :simpleConjug]
parametrisation = POSSIBLE_PARAMS[3]
include("src/fitzHughNagumo.jl")
P = FitzhughDiffusion(10.0, -8.0, 15.0, 0.0, 3.0)
```
Set the starting point:
```julia
x0 = ℝ{2}(-0.5, 0.6)
if parametrisation == :simpleAlter
    x0 = regularToAlter(x0, P.ϵ, 0.0)
elseif parametrisation == :simpleConjug
    x0 = regularToConjug(x0, P.ϵ, 0.0)
end
```
Define the time grid over which the underlying process needs to be simulated:
```julia
dt = 1/50000
T = 10.0
tt = 0.0:dt:T
```
Simulate the path:
```julia
XX, _ = simulateSegment(0.0, x0, P, tt)
```
Define the observational regime (let's take observations of the first coordinate, distanced by 1 time unit):
```julia
skip = 50000
L = @SMatrix [1. 0.]
Time = collect(tt)[1:skip:end]
x1 = [(L*x)[1] for x in XX.yy[1:skip:end]]
x2 = [NaN for t in Time]
x2[1] = x0[2]
```
And finally save the data
```julia
df = DataFrame(time=Time, x1=x1, x2=x2)
CSV.write(outdir*"sparse_path_part_obs_"*String(parametrisation)*".csv", df)
```