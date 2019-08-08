# BridgeSDEInference.jl

MCMC sampler for inference for diffusion processes with the use of Guided Proposals using the package [Bridge.jl](https://github.com/mschauer/Bridge.jl). Currently under development.


## Overview

The main function introduced by this package is
```julia
mcmc(::Type{K}, ::ObsScheme, obs, obsTimes, yPr::StartingPtPrior, w,
     P˟, P̃, Ls, Σs, numSteps, tKernel, priors, τ;
     fpt=fill(NaN, length(obsTimes)-1), ρ=0.0, dt=1/5000, saveIter=NaN,
     verbIter=NaN, updtCoord=(Val((true,)),), paramUpdt=true,
     skipForSave=1, updtType=(MetropolisHastingsUpdt(),),
     blocking::Blocking=NoBlocking(),
     blockingParams=([], 0.1, NoChangePt()),
     solver::ST=Ralston3(), changePt::CP=NoChangePt(), warmUp=0)
```
It finds the posterior distribution of the unknown parameters given discrete time observations of the underlying process. [`Scripts` folder](scripts/) contains a few example scripts which read the data from files, set up the observational scheme, call `mcmc` function and plot the results. We take the file `inference_without_blocking.jl` and explain the steps taken there in more detail below.

## Example (see [inference_without_blocking.jl](scripts/inference_without_blocking.jl) for full code)
### Running the sampler
First set global parameters and load the dependencies:
```julia
SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR = joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
using Main.BridgeSDEInference
using Distributions # to define priors
using Random        # to seed the random number generator
using DataFrames
using CSV
include(joinpath(AUX_DIR, "read_and_write_data.jl"))
include(joinpath(AUX_DIR, "transforms.jl"))
```
We'll be working in a partial observations setting, which means we turn-off the flag about the first-passage time setting
```julia
fptObsFlag = false
```
We can load the data from a file:
```julia
# pick dataset
filename = "path_part_obs_conj.csv"

# fetch the data
(df, x0, obs, obsTime, fpt,
      fptOrPartObs) = readData(Val(fptObsFlag), joinpath(OUT_DIR, filename))
```
To see how data can be generated see [this note](docs/generate_data.md). We assume that the underlying model is given by the FitzHugh-Nagumo diffusion. We choose an appropriate parametrisation of the model (see [this note](docs/parametrisations.md) for more details on available parametrisations)
```julia
param = :complexConjug
```
and take a guess at initial values of the parameters:
```julia
θ₀ = [10.0, -8.0, 15.0, 0.0, 3.0]
```
Let's define now the target and auxiliary laws:
```julia
# Target law
P˟ = FitzhughDiffusion(param, θ₀...)
# Auxiliary law
P̃ = [FitzhughDiffusionAux(param, θ₀..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
     in zip(obsTime[1:end-1], obsTime[2:end], obs[1:end-1], obs[2:end])]
display(P̃[1])
```
The process is two-dimensional and the data is such that we observe its first coordinate without any noise: V=LX, with ![equation](https://latex.codecogs.com/gif.latex?L%3D%281%2C0%29%5ET), at a discrete grid of time-points. For the numerical reasons we assume that we in fact observe V=LX+Z, where Z is Gaussian random variable with mean 0 and miniscule noise. To this end we set observational matrix L and covariance matrix Σ
```julia
L = @SMatrix [1. 0.]
Σdiagel = 10^(-10)
Σ = @SMatrix [Σdiagel]
```
We can now define the observational operator and covariance matrix of the noise at each observation time:
```julia
Ls = [L for _ in P̃]
Σs = [Σ for _ in P̃]
```
We define a time-change function used for numerical purposes and set the number of steps of the Markov chain. We additionally define a convenience number `saveIter`, which says that 1 in every `saveIter` many steps of the MCMC chain the entire accepted path will be saved (and can be later plotted)
```julia
τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
numSteps=1*10^5
saveIter=3*10^2
```
We will be updating `4` coordinates of the vector `θ`. The first three will be completed via conjugate samplers, the last one will be done via Metropolis-Hastings step. First, we define the transition kernel for the Metropolis-Hastings step---we use a random walk. Note that we define `5`-dimensional random walk, despite the fact that not all coordinates are relevant. In particular, we will soon indicate that only the last coordinate of `θ` is supposed to be updated with a Metropolis-Hastings step, consequently, the step-size of the random walk (and information whether respective coordinates need to be kept positive) in any other dimension is irrelevant.
```julia
tKernel = RandomWalk([0.0, 0.0, 0.0, 0.0, 0.5],
                     [false, false, false, false, true])
```
We also specify priors. We choose multivariate normals for conjugate update and an improper prior for the Metropolis-Hastings setp. For more information about convenience functions for priors see [this note](docs/priors.md).
```julia
priors = Priors((MvNormal([0.0,0.0,0.0], diagm(0=>[1000.0, 1000.0, 1000.0])),
                 ImproperPrior()))
```
We set the blocking scheme. For this example we don't want any blocking:
```julia
𝔅 = NoBlocking()
blockingParams = ([], 0.1, NoChangePt())
```
For more information about possible blocking choices, see [this note](docs/blocking.md). We also specify that by default only single ODE solvers for `H`, `Hν` and `c` are to be used:
```julia
changePt = NoChangePt()
```
We specify a prior over the starting point to be a rather uninformative Gaussian:
```julia
x0Pr = GsnStartingPt(x0, x0, @SMatrix [20. 0; 0 20.])
```
For more information about re-sampling of the starting point see [this note](docs/starting_pt.md). Finally, we set the `warmUp` variable, which says for how many initial steps of the MCMC chain no parameter updates need to be made and only path need to be updated. This can sometimes help with silly initialisations of the starting point that can otherwise cause some volatile swaying of the parameter chain in the initial stages of the mcmc chain.
```julia
warmUp = 100
```
We can now run the mcmc sampler:
```julia
Random.seed!(4)
(chain, accRateImp, accRateUpdt,
    paths, time_) = mcmc(eltype(x0), fptOrPartObs, obs, obsTime, x0Pr, 0.0, P˟,
                         P̃, Ls, Σs, numSteps, tKernel, priors, τ;
                         fpt=fpt,
                         ρ=0.975,
                         dt=1/1000,
                         saveIter=saveIter,
                         verbIter=10^2,
                         updtCoord=(Val((true, true, true, false, false)),
                                    Val((false, false, false, false, true)),
                                    ),
                         paramUpdt=true,
                         updtType=(ConjugateUpdt(),
                                   MetropolisHastingsUpdt(),
                                   ),
                         skipForSave=10^0,
                         blocking=𝔅,
                         blockingParams=blockingParams,
                         solver=Vern7(),
                         changePt=changePt,
                         warmUp=warmUp)
```
We passed some additional parameters. `ρ` is the memory parameter for the Cranck-Nicolson scheme. `dt` is the density parameter for the grid on which unobserved parts of the path are imputed. For diagnostic purposes the sampled path is saved once every `saveIter` many steps of the mcmc chain. Once every `verbIter` many steps short info is printed to the console. `updtCoord` is a many-hot encoding, indicating which coordinates of `θ` vector are being updated by a corresponding transition kernel. `updtType`, then gives the type of the update to be performed by the respective transition kernel (`MetropolisHastingsUpdt()` is the most generic update type, `ConjugateUpdt()`---if implemented---allows for sampling from full conditional distributions). The MCMC chain cycles through the entries of `priors.priors`, `updtCoord` and `updtType`, so that the trio of `priors.priors[i]`, `updtCoord[i]`, `updtType[i]` characterises an MCMC update. `paramUpdt` indicates whether parameters need to be updated at all. If not, then only bridges are repeatedly sampled, resulting in `mcmc` function acting as a marginal sampler from the law of the target bridges, conditionally on the parameter values. `skipForSave` is the parameter used to reduce the storage space needed to save the paths---only 1 every `skipForSave` many points of the simulated paths are saved. Finally, `solver` indicates which numerical solver is supposed to be used for solving backward ODEs. The possible choices are: `Ralston3`, `RK4`, `Tsit5`, `Vern7`.

### Inspecting the results
We can inspect acceptance rates:
```julia
print("imputation acceptance rate: ", accRateImp,
      ", parameter update acceptance rate: ", accRateUpdt)
```
We can also save the results to files. For a fair comparison we transform the second coordinate to X under the `:regular` parametrisation.
```julia
x0⁺, pathsToSave = transformMCMCOutput(x0, paths, saveIter; chain=chain,
                                       numGibbsSteps=2,
                                       parametrisation=param,
                                       warmUp=warmUp)

df2 = savePathsToFile(pathsToSave, time_, joinpath(OUT_DIR, "sampled_paths.csv"))
df3 = saveChainToFile(chain, joinpath(OUT_DIR, "chain.csv"))
```
Lastly, we can make some diagnostic plots:
```julia
include(joinpath(AUX_DIR, "plotting_fns.jl"))
set_default_plot_size(30cm, 20cm)
plotPaths(df2, obs=[Float64.(df.x1), [x0⁺[2]]],
          obsTime=[Float64.(df.time), [0.0]], obsCoords=[1,2])

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
