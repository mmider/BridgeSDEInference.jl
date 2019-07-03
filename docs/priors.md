# Priors
The package `MCMCBridge.jl` [defines](../src/priors.jl) some convenience functions for priors. Below we give a brief description of their functionality.

## Motivation

In general, a Markov chain which targets the joint distribution of the unknown parameters and unobserved path: ![equation](https://latex.codecogs.com/gif.latex?%5Cpi%28%5Ctheta%2C%20X%29) may be updating `θ` as a Gibbs sampler, in blocks. In that case, a single `sweep` will consist of a few transition kernel, each of which will be updating a subset of parameters `θ`. For each such transition kernel a prior or priors are needed for the respective coordinates of `θ` that are being updated. This is what the struct `Priors` aims to help with.

## Implementation

`Priors` holds a list of all priors (inside a field `priors`) that are needed by the Markov chain. Additionally, it contains a list of lists called `indicesForUpdt`, where the outer list iterates over transition kernels and each inner list contains the indices of priors that correspond to coordinates of `θ` that are being updated.

## Example 1

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

## Example 2
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