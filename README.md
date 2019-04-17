# MCMCBridge.jl

MCMC sampler for inference for diffusion processes with the use of Guided Proposals. Currently under development. Ultimately the aim is to integrate it into Bridge.jl package.


# Overview

The main function introduced by this package is
```julia
mcmc(::ObsScheme, obs, obsTimes, y, w, P˟, P̃, Ls, Σs, numSteps,
     tKernel, priors, τ; fpt=fill(NaN, length(obsTimes)-1), ρ=0.0,
     dt=1/5000, saveIter=NaN, verbIter=NaN,
     updtCoord=(Val((true,)),), paramUpdt=true,
     skipForSave=1, updtType=(MetropolisHastingsUpdt(),),
     solver::ST=Ralston3())
```
It finds the posterior distribution of the unknown parameters given the discrete time observations of the underlying process. The file `main.jl` contains an example of a script which reads the data from the file, sets up the observational scheme, calls `mcmc` function and plots the results. The same steps but in greater detail are recounted below.

## Example

First load the dependencies:
```julia
outdir="output"

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models # possibly not needed
using DataFrames
using CSV
```
Suppose we are interested in a two-dimensional diffusions X and that we only observe its first coordinate without noise: V=LX, with ![equation](https://latex.codecogs.com/gif.latex?L%3D%281%2C0%29%5ET). For the numerical reasons we assume that we in fact observe V=LX+Z, where Z is Gaussian random variable with mean 0 and miniscule noise. We now set observational matrix L and covariance matrix Σ.
```julia
L = @SMatrix [1. 0.]
Σdiagel = 10^(-10)
Σ = @SMatrix [Σdiagel]
```
We can also load the model. We will assume that the underlying has been generated using FitzHugh-Nagumo model. We implemented a couple of different parametrisations of this model:
```
POSSIBLE_PARAMS = [:regular, :simpleAlter, :complexAlter, :simpleConjug,
                   :complexConjug]
parametrisation = POSSIBLE_PARAMS[5]
```
`:regular` parametrisation is the most commonly encountered parametrisation of the FitzHugh-Nagumo model:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20dY_t%26%3D%5Cfrac%7B1%7D%7B%5Cepsilon%7D%5Cleft%28Y_t-Y_t%5E3-X_t%20&plus;%20s%5Cright%29dt%5C%5C%20dX_t%26%3D%5Cleft%28%5Cgamma%20Y_t%20-%20X_t%20&plus;%5Cbeta%5Cright%29dt%20&plus;%20%5Csigma%20dW_t%2C%20%5Cend%7Balign*%7D)

The proposal is a Guided Proposal with auxiliary law ![equation](https://latex.codecogs.com/gif.latex?%5Cwidetilde%7BP%7D) induced by the linear diffusion obtained by linearising FitzHugh-Nagumo diffusion at an end-point.

`:simpleAlter` and `:complexAlter` are re-parametrised forms of FitzHugh-Nagumo model, in which the first coordinate is simply an integrated second coordinate:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20dY_t%20%26%3D%20%5Cdot%7BY%7D_tdt%5C%5C%20d%5Cdot%7BY%7D_t%20%26%3D%20%5Cfrac%7B1%7D%7B%5Cepsilon%7D%5Cleft%28%281-%5Cgamma%29Y_t%20-Y_t%5E3%20-%5Cepsilon%20%5Cdot%7BY%7D_t%20&plus;s-%5Cbeta%20&plus;%20%281-3Y_t%5E2%29%5Cdot%7BY%7D_t%20%5Cright%20%29dt%20&plus;%20%5Cfrac%7B%5Csigma%7D%7B%5Cepsilon%7DdW_t%20%5Cend%7Balign*%7D)

The difference between the two parametrisations manifests itself with the different choices of auxiliary diffusions. For the former it is a pair of integrated Brownian motion and a Brownian motion. For the latter it is a two dimensional diffusion, where the second coordinate is a linear diffusion obtained from linearising ![equation](https://latex.codecogs.com/gif.latex?%5Cdot%7BY%7D) at an end-point and the first coordinate is an integrated second coordinate.

Finally, `:simpleConjug` and `:complexConjug` are defined analogously to `:simpleAlter` and `:complexAlter`, with an exception that an additional step of re-defining the parameters:

![equation](https://latex.codecogs.com/gif.latex?s%5Cleftarrow%20%5Cfrac%7Bs%7D%7B%5Cepsilon%7D%2C%5Cquad%20%5Cbeta%5Cleftarrow%20%5Cfrac%7B%5Cbeta%7D%7B%5Cepsilon%7D%2C%5Cquad%20%5Csigma%5Cleftarrow%20%5Cfrac%7B%5Csigma%7D%7B%5Cepsilon%7D%2C%5Cquad%20%5Cgamma%5Cleftarrow%20%5Cfrac%7B%5Cgamma%7D%7B%5Cepsilon%7D%2C%5Cquad%20%5Cepsilon%5Cleftarrow%20%5Cfrac%7B1%7D%7B%5Cepsilon%7D%2C)

is made. We chose the `complexConjug` parametrisation. Now we can load the model and the `MCMCBridge.jl` package
```julia
include("src/fitzHughNagumo.jl")
include("src/fitzHughNagumo_conjugateUpdt.jl")
include("src/MCMCBridge.jl")
using Main.MCMCBridge
```

We define two functions for loading the data, depending on whether inference is supposed to be done on partially observed diffusions or first-passage time observations:
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
(df, x0, obs, obsTime, fpt,
    fptOrPartObs) = readData(Val(fptObsFlag),
                             joinpath(outdir,"path_part_obs_conj.csv"))
```
The first-passage time observation regime is not fully tested and falls outside of the scope of this project so might be removed in future iterations. The data can be generated easily with the code written in the file `simulate_data.jl`.

We can now set the initial guess for parameter θ₀:
```julia
θ₀ = [10.0, -8.0, 15.0, 0.0, 3.0]
```
Define the target and auxiliary law:
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
The time-change function used for numerical purposes:
```julia
τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
```
Set the number of steps and define the transition kernel for the Metropolis-Hastings step (notice that even though we specify the range for Uniform random variates updating each coordinate of the vector θ, we will be in fact updating only some of those coordinates with the Metropolis-Hastings steps and all the other coordinate updates will not be making use of this transition kernel, we will specify which coordinates will need updates with this kernel later on)
```julia
numSteps=3*10^4
tKernel=RandomWalk(θ₀, [3.0, 5.0, 5.0, 0.01, 0.5],
                   [false, false, false, false, true])
```
Finally, we specify the priors:
```julia
priors = ((MvNormal([0.0,0.0,0.0],
                    diagm(0=>[1000.0, 1000.0, 1000.0])),),
          #(ImproperPrior(),),
          #(ImproperPrior(),),
          #(ImproperPrior(),),
          (ImproperPrior(),),
          )
```
These are supposed to be in a format: ((prior11, prior12, ...), (prior21, prior22, ...), ...). The parameter updates are done in a cycling fashion, so that first (prior11, prior12, ...) are used at first iteration of MCMC, in the second step (prior21, prior22, ...) are used etc. until all priors in the prior's vector are exhausted and then in the next mcmc step ((prior11, prior12, ...) are used again and the cycle is repeated. Each container (priori1, priori2, ...) must contain all relevant priors for a given MCMC update and in case the update samples from full-conditionals, the conjugate prior must be on the first place, i.e. it must be priori1.

We can now run the mcmc sampler:
```julia
Random.seed!(4)
(chain, accRateImp, accRateUpdt,
    paths, time_) = mcmc(fptOrPartObs, obs, obsTime, x0, 0.0, P˟, P̃, Ls, Σs,
                         numSteps, tKernel, priors, τ;
                         fpt=fpt,
                         ρ=0.995,
                         dt=1/5000,
                         saveIter=3*10^2,
                         verbIter=10^2,
                         updtCoord=(Val((true, true, true, false, false)),
                                    #Val((true, false, false, false, false)),
                                    #Val((false, true, false, false, false)),
                                    #Val((false, false, true, false, false)),
                                    Val((false, false, false, false, true)),
                                    ),
                         paramUpdt=true,
                         updtType=(ConjugateUpdt(),
                                   #MetropolisHastingsUpdt(),
                                   #MetropolisHastingsUpdt(),
                                   #MetropolisHastingsUpdt(),
                                   MetropolisHastingsUpdt(),
                                   ),
                         skipForSave=10^1,
                         solver=Vern7())
```
We passed some additional parameters. `ρ` is the memory parameter for the Cranck-Nicolson scheme. `dt` is the density parameter for the grid on which unobserved parts of the path are imputed. For diagnostic purposes the sampled path is save once every `saveIter` many steps of the mcmc chain. Once every `verbIter` many steps short info is printed to a console. `updtCoord` and `updtType` are both in an anlogous format to `priors`. The mcmc chain cycles through entries of `priors`, `updtCoord` and `updtType`, so that the trio of `priors[i]`, `updtCoord[i]`, `updtType[i]` characterises an mcmc update. `updtCoord` indicates which coordinates of a parameter vector are supposed to be updated in a given step, whereas `updtType` differentiates between which type of update is supposed to be applied. `MetropolisHastingsUpdt()` is the most generic update type, however if implemented, `ConjugateUpdt()` allows for sampling from full conditional distributions. `paramUpdt` indicates whether parameters need to be updated at all. If not, then only bridges are repeatedly sampled, resulting in a marginal sampler from the law of target bridges, conditionally on the parameter values. `skipForSave` is the parameter used to reduce the storage space needed to save the paths---only 1 every `skipForSave` many points of the simulated paths are saved. Finally, `solver` indicates which numerical solver is supposed to be used for solving the backward ODEs. The possible choices are: `Ralston3`, `RK4`, `Tsit5`, `Vern7`.

We can inspect the acceptance rates:
```julia
print("imputation acceptance rate: ", accRateImp,
      ", parameter update acceptance rate: ", accRateUpdt)
```

We can also save the results to files. For a fair comparison we transform the second coordinate to X under the `:regular` parametrisation.
```julia
if parametrisation in (:simpleAlter, :complexAlter)
    pathsToSave = [[alterToRegular(e, θ[1], θ[2]) for e in path] for (path,θ)
                                      in zip(paths, chain[1:3*10^2:end][2:end])]
    # only one out of many starting points will be drawn
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
