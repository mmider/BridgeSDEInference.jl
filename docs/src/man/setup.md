# Defining the observational scheme and parameters of the Markov chain

There is a series of objects that define the Markov chain Monte Carlo sampler
and that the user needs to define in order to be able to run the inference
algorithm. To keep this processes structured an object `MCMCSetup` is defined.
It allows for a systematic and concise way of defining the MCMC sampler.


## Defining the processes
To define the `MCMCSetup` one needs to decide on the `Target` diffusion an
`Auxiliary` diffusion and the observation scheme. For instance, suppose that
there are three observations:
```julia
obs = [...]
obs_times = [1.0, 2.0, 3.0]
```
Then the `target` diffusion is defined globally and the `auxiliary` diffusion
is defined separately on each interval
```julia
P_target = TargetDiffusion(parameters)
P_auxiliary = [AuxiliaryDiffusion(parameters, o, t) for (o,t) in zip(obs, obs_times)]
```
To define the setup for partially observed diffusion it is enough to write:
```julia
setup = MCMCSetup(P_target, P_auxiliary, PartObs()) # for first passage times use FPT()
```
```@docs
MCMCSetup
```

## Observations
To set the observations, apart from passing the observations and observation
times it is necessary to pass the observational operators and as well as
covariance of the noise. Additionally, one can pass additional information
about the first-passage scheme [TO DO add details on fpt].
```julia
L = ...
Σ = ...
set_observations!(setup, [L for _ in obs], [Σ for _ in obs], obs, obs_time)
```
```@docs
set_observations!
```

## Imputation grid
There are two objects that define the imputation grid. The time step `dt` and
the time transformation that transforms a regular time-grid with equidistantly
distributed imputation points. The second defaults to a usual transformation
employed in papers on the guided proposals. [TO DO add also space-time transformation from the original paper for the bridges]. It is enough to call
```julia
dt = ...
set_imputation_grid!(setup, dt)
```
```@docs
set_imputation_grid!
```

## Transition kernels
To define the updates of the parameters and the Wiener path a couple of objects
need to specified. A boolean flag needs to be passed indicating whether any
parameter updates are to be performed. If set to `false` then only the path
is updated and the result it a marginal sampler on a path space. Additionally
a memory (or persistence) parameter of the preconditioned Crank-Nicolson scheme
needs to be set for the path updates. For the parameter updates three
additional objects must be specified. A sequence of transition kernels---one
for each Gibbs step, a sequence of lists indicating parameters to be
updated---one list for each Gibbs step and a sequence of indicators about the
types of parameter updates---one for each Gibbs step. Additionally, an object
describing an adaptation scheme for the auxiliary law can be passed [TODO add
description to the last]
### Random walk
The package provides an implementation of a `random walk`, which can be used
as a generic transition kernel
```@docs
RandomWalk
```
### Indicators for parameter update
The indicators for parameter updates should be in a format of tuple of tuples
(or arrays of arrays etc.). Each inner tuple corresponds to a single Gibbs step
and the elements of the inner tuples give indices of parameters that are to be
updated on a given Gibbs step. For instance: `((1,2,3),(5,))` says that in the
first Gibbs step the first three parameters are to be updated, whereas in the
second Gibbs step parameter `5` is to be updated.

### Flags for the types of parameter updates
There are currently two different ways of updating parameters:
```@docs
ConjugateUpdt
MetropolisHastingsUpdt
```

### Setting transition kernels
An example of setting the transition kernels is as follows:
```julia
pCN = ... # memory paramter of the preconditioned Crank Nicolson scheme
update_parameters = true
set_transition_kernels!(setup,
                        [RandomWalk([],[]),
                         RandomWalk([3.0, 5.0, 5.0, 0.01, 0.5], 5)],
                        pCN, update_parameters, ((1,2,3),(5,)),
                        (ConjugateUpdt(),
                         MetropolisHastingsUpdt(),
                        ))
```

```@docs
set_transition_kernels!
```


## Priors
The package `BridgeSDEInference.jl` [defines](../src/priors.jl) some convenience functions for priors. Below we give a brief description of their functionality.
```@docs
```

### Motivation

In general, a Markov chain which targets the joint distribution of the unknown parameters and the unobserved path: ![equation](https://latex.codecogs.com/gif.latex?%5Cpi%28%5Ctheta%2C%20X%29) may be updating `θ` by employing a Gibbs sampler, effectively updating sections of `θ` at a time. In that case, a sequence of transition kernels are applied to `θ` over and over again: K₁,K₂,...,Kₖ,K₁,K₂,.... We refer to one such complete sequence of transition kernels (K₁,K₂,...,Kₖ) as a single `sweep`. There is a single prior that the parameter `θ` is equipped with, but due to the nature of updates it is useful to map this prior to respective transition kernels. This is done by considering coordinates of `θ` that are being updated by respective transition kernel, taking only a prior over those coordinates and associate it with the respective transition kernel. This mapping is what the struct `Priors` aims to help with.

### Implementation

`Priors` holds a list of all priors (inside a field `priors`) that are needed by the Markov chain. Additionally, it contains a list of lists called `indicesForUpdt`, where the outer list iterates over transition kernels in a `sweep` and where each inner list contains the indices of priors that correspond to coordinates of `θ` that are being updated by a corresponding transition kernel.

### Example 1

This is perhaps the most common use case. Suppose that `n` coordinates---where we take `n=3` for illustration purposes---of prameter `θ` are being updated by the Markov chain. Suppose further that there are `n` transition kernels, each updating a separate coordinate. Let's assume for simplicity that each coordinate is equipped with an independent, improper prior. Then the `Priors` struct can be set up as follows:
```julia
include("src/priors.jl")
include("src/types.jl")
n = 3
priors = Priors([ImproperPrior() for i in 1:n])

display(priors.priors)
(ImproperPrior(), ImproperPrior(), ImproperPrior())

display(priors.indicesForUpdt)
3-element Array{Array{Int64,1},1}:
 [1]
 [2]
 [3]
```
Notice that only a list of priors had to be supplied and the struct took care of setting up the `indicesForUpdt` container.

### Example 2
Suppose that the Markov chain uses two transition kernels. The first transition kernel updates `3` coordinates of `θ`, which have a corresponding joint, multivariate Normal prior with some pre-specified covariance matrix `Σ` and mean `0`. The second transition kernel updates `2` coordinates of `θ`, and these two are equipped with independent, improper priors. Then the `Priors` struct can be set up as follows:
```julia
include("src/priors.jl")
include("src/types.jl")
using LinearAlgebra, Distributions
Σ = diagm(0=>[1000.0, 1000.0, 1000.0])
μ = [0.0,0.0,0.0]
priors = Priors([MvNormal(μ, Σ), ImproperPrior(), ImproperPrior()],
                [[1],[2,3]])
display(priors.priors)
(FullNormal(
dim: 3
μ: [0.0, 0.0, 0.0]
Σ: [1000.0 0.0 0.0; 0.0 1000.0 0.0; 0.0 0.0 1000.0]
)
, ImproperPrior(), ImproperPrior())

display(priors.indicesForUpdt)
2-element Array{Array{Int64,1},1}:
 [1]   
 [2, 3]
```
Notice that we had to pass all priors in a list and then specify that the first transition kernel uses only the first prior, whereas the second transition kernel uses two priors from the list.




## Re-sampling of the starting point
There are two options for specifying the starting point.

- It is either known and fixed to some value, in which case the following should be passed to `mcmc` function:
```julia
x0Pr = KnownStartingPt(x0)
```
where `x0` is the (vector-)value of the starting point. `KnownStartingPt` is equivalent to a discrete prior which puts all its mass on a single point.

- Alternatively, a Gaussian prior can be specified via `GsnStartingPt`, for instance:
```julia
Σₛₚ = @SMatrix [20. 0; 0 20.]
μₛₚ = @SVector[1.0, 2.0]
x₀ = @SVector[2.0, 3.0]
x0Pr = GsnStartingPt(x₀, μₛₚ, Σₛₚ)
```
to specify a Gaussian prior with mean μₛₚ and covariance matrix Σₛₚ and initialise the guess for a starting point to x₀. Alternatively, x₀ can be omitted, in which case the starting point will be sampled randomly, according the prior distribution:
```julia
x0Pr = GsnStartingPt(μₛₚ, Σₛₚ)
```



## Blocking
Currently two choices of blocking are available:
* No blocking at all, in which case
```julia
𝔅 = NoBlocking()
blockingParams = ([], 0.1, NoChangePt())
```
should be set
* Blocking using chequerboard updating scheme. For this updating scheme, at each observation a knot can be (but does not have to be) placed. IMPORTANT: The knot indexing starts at the first non-starting point observation. Suppose we have, say, `20` observations (excluding the starting point). Let's put a knot on every other observation, ending up with knots on observations with indices: `[2,4,6,8,10,12,14,16,18,20]`. Chequerboard updating scheme splits these knots into two, disjoint, interlaced subsets, i.e. `[2,6,10,14,18]` and `[4,8,12,16,20]`. This also splits the path into two interlaced sets of blocks: `[1–2,3–6,7–10,11–14,15–18,19–20]`, `[1–4,5–8,9–12,13–16,17–20]` (where interval indexing starts with interval 1, whose end-points are the starting point and the first non-starting point observation). The path is updated in blocks. First, blocks `[1–2,3–6,7–10,11–14,15–18,19–20]` are updated conditionally on full and exact observations indexed with `[2,6,10,14,18]`, as well as all the remaining, partial observations (indexed by `[1,2,3,...,20]`). Then, the other set of blocks is updated in the same manner. This is then repeated. To define the blocking behaviour, only the following needs to be written:
```julia
𝔅 = ChequeredBlocking()
blockingParams = (collect(2:20)[1:2:end], 10^(-10), SimpleChangePt(100))
```
The first defines the blocking updating scheme (in the future there might be a larger choice). The second line places the knots on `[2,4,6,8,10,12,14,16,18,20]`. Splitting into appropriate subsets is done internally. `10^(-10)` is an artificial noise parameter that needs to be added for the numerical purposes. Ideally we want this to be as small as possible, however the algorithm may have problems with dealing with very small values. The last arguments aims to remedy this. `SimpleChangePt(100)` has two functions. One, it is a flag to the `mcmc` sampler that two sets of ODE solvers need to be employed: for the segment directly adjacent to a knot from the left ODE solvers for `M⁺`, `L`, `μ` are employed and `H`, `Hν` and `c` are computed as a by-product. On the remaining part of blocks, the ODE solvers for `H`, `Hν` and `c` are used directly. The second function is to pass the point at which a change needs to be made between these two solvers (which for the example above is set to `100`). The reason for this functionality is that solvers for `M⁺`, `L`, `μ` are more tolerant to very small values of the artificial noise.
