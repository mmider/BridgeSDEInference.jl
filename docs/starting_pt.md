[back to README](../README.md)
# Re-sampling of the starting point
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