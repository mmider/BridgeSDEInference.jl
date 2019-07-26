[back to README](../README.md)
# Priors
The package `BridgeSDEInference.jl` [defines](../src/priors.jl) some convenience functions for priors. Below we give a brief description of their functionality.

## Motivation

In general, a Markov chain which targets the joint distribution of the unknown parameters and the unobserved path: ![equation](https://latex.codecogs.com/gif.latex?%5Cpi%28%5Ctheta%2C%20X%29) may be updating `θ` by employing a Gibbs sampler, effectively updating sections of `θ` at a time. In that case, a sequence of transition kernels are applied to `θ` over and over again: K₁,K₂,...,Kₖ,K₁,K₂,.... We refer to one such complete sequence of transition kernels (K₁,K₂,...,Kₖ) as a single `sweep`. There is a single prior that the parameter `θ` is equipped with, but due to the nature of updates it is useful to map this prior to respective transition kernels. This is done by considering coordinates of `θ` that are being updated by respective transition kernel, taking only a prior over those coordinates and associate it with the respective transition kernel. This mapping is what the struct `Priors` aims to help with.

## Implementation

`Priors` holds a list of all priors (inside a field `priors`) that are needed by the Markov chain. Additionally, it contains a list of lists called `indicesForUpdt`, where the outer list iterates over transition kernels in a `sweep` and where each inner list contains the indices of priors that correspond to coordinates of `θ` that are being updated by a corresponding transition kernel.

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