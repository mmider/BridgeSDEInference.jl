[back to README](../README.md)
# Blocking
Currently two choices of blocking are available:
* No blocking at all, in which case
```julia
𝔅 = NoBlocking()
blockingParams = ([], 0.1)
```
should be set
* Blocking using chequerboard updating scheme. For this updating scheme, at each observation a knot can be (but does not need to be) placed. Suppose we have, say, `20` observations (excluding the starting point). Let's put a knot on every other observation, ending up with knots on observations with indices: `[2,4,6,8,10,12,14,16,18,20]`. Chequerboard updating scheme splits these knots into two, disjoint, interlaced subsets, i.e. `[2,6,10,14,18]` and `[4,8,12,16,20]`. This also splits the path into two interlaced sets of blocks: `[1-2,3-6,7-10,11-14,15-18,19-20]`, `[1-4,5-8,9-12,13-16,17-20]`, where the number indicate interval indices. The path is updated in blocks. First, blocks `[1-2,3-6,7-10,11-14,15-18,19-20]` are updated conditionally on full and exact observations indexed with `[2,6,10,14,18]`, as well as all the remaining, partial observations (indexed by `[1,2,3,...,20]`). Then, the other set of blocks is updated in the same manner. This is then repeated. To define the blocking behaviour, only the following needs to be written:
```julia
𝔅 = ChequeredBlocking()
blockingParams = (collect(2:20)[1:2:end], 10^(-6))
```
The first defines the blocking updating scheme (in the future there might be a larger choice). The second line places the knots on `[2,4,6,8,10,12,14,16,18,20]`. The splitting into appropriate subsets is done internally. `10^(-6)` is an artificial noise parameter that needs to be added for the numerical purposes. Ideally we want this to be as small as possible, however currently the algorithm sometimes has problems with dealing with very small values and some improvements to numerical ODE solvers are needed.

