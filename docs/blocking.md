[back to README](../README.md)
# Blocking
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

