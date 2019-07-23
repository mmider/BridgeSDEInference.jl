# BridgeSDEInference.jl

MCMC sampler for inference for diffusion processes with the use of Guided Proposals. Currently under development. Ultimately the aim is to integrate it into Bridge.jl package.


## Overview

The main function introduced by this package is
```julia
mcmc(::Type{K}, ::ObsScheme, obs, obsTimes, y, w, P˟, P̃, Ls, Σs, numSteps,
     tKernel, priors, τ; fpt=fill(NaN, length(obsTimes)-1), ρ=0.0,
     dt=1/5000, saveIter=NaN, verbIter=NaN,
     updtCoord=(Val((true,)),), paramUpdt=true,
     skipForSave=1, updtType=(MetropolisHastingsUpdt(),),
     blocking::Blocking=NoBlocking(), blockingParams=([], 0.1),
     solver::ST=Ralston3())
```
It finds the posterior distribution of the unknown parameters given discrete time observations of the underlying process. The file `main.jl` contains an example of a script which reads the data from the file, sets up the observational scheme, calls `mcmc` function and plots the results. The same steps but in greater detail are recounted below.

## Example (see main.jl for full code)
### Running the sampler
First load the dependencies:
```julia
mkpath("output/")
outdir="output"

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using DataFrames
using CSV
```
Suppose we are interested in a two-dimensional diffusions X and that we only observe its first coordinate without noise: V=LX, with ![equation](https://latex.codecogs.com/gif.latex?L%3D%281%2C0%29%5ET). For the numerical reasons we assume that we in fact observe V=LX+Z, where Z is Gaussian random variable with mean 0 and miniscule noise. To this end we set observational matrix L and covariance matrix Σ
```julia
L = @SMatrix [1. 0.]
Σdiagel = 10^(-10)
Σ = @SMatrix [Σdiagel]
```
We can also load the FitzHugh-Nagumo model. `5` different parametrisations of this model have been implemented in this package---please see [this note](docs/parametrisations.md) for more details on the differences between them. We will use parametrisation `:complexConjug`. We can also load the `MCMCBridge.jl` package
```julia
POSSIBLE_PARAMS = [:regular, :simpleAlter, :complexAlter, :simpleConjug,
                   :complexConjug]
parametrisation = POSSIBLE_PARAMS[5]
include("src/fitzHughNagumo.jl")
include("src/fitzHughNagumo_conjugateUpdt.jl")
include("src/MCMCBridge.jl")
using Main.MCMCBridge
```
We define two functions for loading the data, depending on whether inference is supposed to be done on partially observed diffusions or first-passage time observations. Let's pick the former
```julia
function readData(::Val{true}, filename)
    df = CSV.read(filename)
    x0 = ℝ{2}(df.upCross[1], df.x2[1])
    obs = ℝ{1}.(df.upCross)
    obsTime = Float64.(df.time)
    fpt = [FPTInfo((1,), (true,), (resetLvl,), (i==1,)) for
            (i, resetLvl) in enumerate(df.downCross[2:end])]
    fptOrPartObs = FPT()
    df, x0, obs, obsTime, fpt, fptOrPartObs
end

function readData(::Val{false}, filename)
    df = CSV.read(filename)
    obs = ℝ{1}.(df.x1)
    obsTime = Float64.(df.time)
    x0 = ℝ{2}(df.x1[1], df.x2[1])
    fpt = [NaN for _ in obsTime[2:end]]
    fptOrPartObs = PartObs()
    df, x0, obs, obsTime, fpt, fptOrPartObs
end

# decide if first passage time observations or partially observed diffusion
fptObsFlag = false
if fptObsFlag
    filename = "up_crossing_times_regular.csv"
else
    filename = "path_part_obs_conj.csv"
end
(df, x0, obs, obsTime, fpt,
    fptOrPartObs) = readData(Val(fptObsFlag),
                             joinpath(outdir, filename))
```
To see how data can be generated see [this note](docs/generate_data.md). Let's set the initial guess for the parameter `θ₀`:
```julia
θ₀ = [10.0, -8.0, 15.0, 0.0, 3.0]
```
define the target and auxiliary laws
```julia
# Target law
P˟ = FitzhughDiffusion(θ₀...)
# Auxiliary law
P̃ = [FitzhughDiffusionAux(θ₀..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
      in zip(obsTime[1:end-1], obsTime[2:end], obs[1:end-1], obs[2:end])]
```
Define the observational operator and covariance matrix of the noise at each observation time:
```julia
Ls = [L for _ in P̃]
Σs = [Σ for _ in P̃]
```
The time-change function used for numerical purposes and set the number of steps of the Markov chain:
```julia
τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
numSteps=3*10^4
```
We will be updating `4` coordinates of the vector `θ`. The first three will be completed via conjugate samplers, the last one will be done via Metropolis-Hastings step. First, we define the transition kernel for the Metropolis-Hastings step---we use a random walk. Note that we define `5`-dimensional random walk, despite the fact that not all coordinates are relevant. In particular, we will soon indicate that only the last coordinate of `θ` is supposed to be updated with a Metropolis-Hastings step, consequently, the step-size of the random walk (and info whether respective coordinates need to be kept positive) in any other dimension is irrelevant.
```julia
tKernel = RandomWalk([0.0, 0.0, 0.0, 0.0, 0.5],
                     [false, false, false, false, true])
```
We also specify priors. We choose multivariate normals for conjugate update and an improper prior for the Metropolis-Hastings setp. For more information about convenience functions for priors see [this note](docs/priors.md).
```julia
priors = Priors((MvNormal([0.0,0.0,0.0], diagm(0=>[1000.0, 1000.0, 1000.0])),
                 ImproperPrior()))
```
Finally, we set the blocking scheme. For this example we don't want any blocking:
```julia
𝔅 = NoBlocking()
blockingParams = ([], 0.1)
```
For more information about possible blocking choices, see [this note](docs/blocking.md)
We can now run the mcmc sampler:
```julia
Random.seed!(4)
(chain, accRateImp, accRateUpdt,
    paths, time_) = mcmc(eltype(x0), fptOrPartObs, obs, obsTime, x0, 0.0, P˟, P̃, Ls, Σs,
                         numSteps, tKernel, priors, τ;
                         fpt=fpt,
                         ρ=0.975,
                         dt=1/10000,
                         saveIter=3*10^2,
                         verbIter=10^2,
                         updtCoord=(Val((true, true, true, false, false)),
                                    Val((false, false, false, false, true)),
                                    ),
                         paramUpdt=true,
                         updtType=(ConjugateUpdt(),
                                   MetropolisHastingsUpdt(),
                                   ),
                         skipForSave=10^1,
                         blocking=𝔅,
                         blockingParams=blockingParams,
                         solver=Vern7())
```
We passed some additional parameters. `ρ` is the memory parameter for the Cranck-Nicolson scheme. `dt` is the density parameter for the grid on which unobserved parts of the path are imputed. For diagnostic purposes the sampled path is saved once every `saveIter` many steps of the mcmc chain. Once every `verbIter` many steps short info is printed to the console. `updtCoord` is a many-hot encoding, indicating which coordinates of `θ` vector are being updated by a corresponding transition kernel. `updtType`, then gives the type of the update to be performed by the respective transition kernel (`MetropolisHastingsUpdt()` is the most generic update type, `ConjugateUpdt()`---if implemented---allows for sampling from full conditional distributions). The mcmc chain cycles through the entries of `priors.priors`, `updtCoord` and `updtType`, so that the trio of `priors.priors[i]`, `updtCoord[i]`, `updtType[i]` characterises an mcmc update. `paramUpdt` indicates whether parameters need to be updated at all. If not, then only bridges are repeatedly sampled, resulting in `mcmc` function acting as a marginal sampler from the law of target bridges, conditionally on the parameter values. `skipForSave` is the parameter used to reduce the storage space needed to save the paths---only 1 every `skipForSave` many points of the simulated paths are saved. Finally, `solver` indicates which numerical solver is supposed to be used for solving backward ODEs. The possible choices are: `Ralston3`, `RK4`, `Tsit5`, `Vern7`.

### Inspecting the results
We can inspect acceptance rates:
```julia
print("imputation acceptance rate: ", accRateImp,
      ", parameter update acceptance rate: ", accRateUpdt)
```
We can also save the results to files. For a fair comparison we transform the second coordinate to X under the `:regular` parametrisation.
```julia
if parametrisation in (:simpleAlter, :complexAlter)
    pathsToSave = [[alterToRegular(e, θ[1], θ[2]) for e in path] for (path,θ)
                                      in zip(paths, chain[1:3*10^2:end][2:end])]
    # only one out of many starting points will be plotted
    x0 = alterToRegular(x0, chain[1][1], chain[1][2])
elseif parametrisation in (:simpleConjug, :complexConjug)
    pathsToSave = [[conjugToRegular(e, θ[1], 0) for e in path] for (path,θ)
                                      in zip(paths, chain[1:3*10^2:end][2:end])]
    x0 = conjugToRegular(x0, chain[1][1], 0)
end

df2 = savePathsToFile(pathsToSave, time_, joinpath(outdir, "sampled_paths.csv"))
df3 = saveChainToFile(chain, joinpath(outdir, "chain.csv"))
```
Lastly, we can make some diagnostic plots:
```julia
include("src/plots.jl")
# make some plots
set_default_plot_size(30cm, 20cm)
if fptObsFlag
    plotPaths(df2, obs=[Float64.(df.upCross), [x0[2]]],
              obsTime=[Float64.(df.time), [0.0]],obsCoords=[1,2])
else
    plotPaths(df2, obs=[Float64.(df.x1), [x0[2]]],
              obsTime=[Float64.(df.time), [0.0]],obsCoords=[1,2])
end
plotChain(df3, coords=[1])
plotChain(df3, coords=[2])
plotChain(df3, coords=[3])
plotChain(df3, coords=[5])
```

Here are the results, the sampled paths:

![temp](assets/paths.js.svg)

And the Markov chains, for parameter ϵ:

![temp](assets/param1.js.svg)

parameter s:

![temp](assets/param2.js.svg)

parameter γ:

![temp](assets/param3.js.svg)

and parameter σ:

![temp](assets/param5.js.svg)