[back to README](../README.md)
# Generation of data
There are two short scripts for generating data, available [here](../scripts/simulate_part_obs_save_to_csv.jl) and [here](../scripts/simulate_fpt_save_to_csv.jl). The former generates data for the setting of a partially observed process, the latter is for the setting of the first passage time observations. In both, the law of the target process is controlled (up to some differences in the values of the parameters) by 
```julia
param = :simpleConjug
P = FitzhughDiffusion(param, 10.0, -8.0, 15.0, 0.0, 3.0)
```
The starting point needs to be set and transformed to appropriate parametrisation :
```julia
x0 = ℝ{2}(-0.5, 0.6) # in regular parametrisation
x0 = regularToConjug(x0, P.ϵ, 0.0) # translate to conjugate parametrisation
```
## Partially observed diffusion
Define the time grid over which the underlying process needs to be simulated:
```julia
dt = 1/50000
T = 10.0
tt = 0.0:dt:T
```
And simulate the path
```julia
Random.seed!(4)
 XX, _ = simulateSegment(0.0, x0, P, tt)
```
The remaining lines:
```julia
num_obs = 100
skip = div(length(tt), num_obs)
Time = collect(tt)[1:skip:end]
df = DataFrame(time=Time, x1=[x[1] for x in XX.yy[1:skip:end]],
               x2=[(i==1 ? x0[2] : NaN) for (i,t) in enumerate(Time)])
```
simply define how the process is observed: the distance between recorded observations as well as
which coordinates and the nature of their perturbation (in this example they are not perturbed at all and only the first coordinate is observed). Finally the data can be saved
```julia
CSV.write(FILENAME_OUT, df)
```
## First passage time observations
In this case, `T` in the time grid defines the length of a single segment over which the path is simulated. The simulation needs to be broken down into pieces, because for large overall `T` the path might take up more memory than a computer can handle. `N` defines the number of segments that need to be simulated and pieced together. The following two:
```julia
upLvl = 0.5
downLvl = -0.5
```
specify the up-crossing level and the down-crossing (reset) level. Then, in line 32:
```julia
recentlyUpSearch = true
```
it says that at the time that the process starts it is assumed that the down-crossing has already occurred. If
```julia
recentlyUpSearch = false
```
then the process would have first needed to reach level `downLvl`, before the first time of reaching `upLvl` would be counted. The remaining lines simply go through with the simulation required for this observation setting and finally save the data in
```julia
CSV.write(FILENAME_OUT, df)
```