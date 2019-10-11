# Definition of a diffusion process
To define a diffusion model suitable for inference with `BridgeSDEInference.jl`
one must define two processes:
- the `target` diffusion
- and the `auxiliary` diffusion


## Imports
The diffusion definitions will extend some of the functionality of the existing
functions from `Bridge.jl`, so the following imports are necessary:
```julia
using Bridge
import Bridge: b, σ, B, β, a, constdiff
```

## Definition of the target process
The target process needs to be a struct inheriting from
`ContinuousTimeProcess{ℝ{d,T}}` (where `d` is a dimension of the diffusion) with
the first type parameter `{T}` defining the data-type of the parameters. The
members of the struct must define the diffusion process and are usually limited
to a list of parameters. An example of a valid definition of a `10` dimensional
diffusion with two parameters ``\alpha`` and ``\beta`` would be
```julia
struct TargetDiffusion{T} <: ContinuousTimeProcess{ℝ{10,T}}
    α::T
    β::T
    TargetDiffusion(α::T, β::T) where T = new{T}(α, β)
end
```
Then, one must specify the dynamics of the diffusion by specifying the behaviour
of the drift and volatility functions `b` and `σ` (defined in `Bridge.jl`) for
`TargetDiffusion`:
```julia
b(t, x, P::TargetDiffusion) = foo(t, x, P.α, P.β)
σ(t, x, P::TargetDiffusion) = bar(t, x, P.α, P.β)
```
where `foo` and `bar` are some user-defined functions. Finally, three auxiliary
functions must be defined:
```julia
constdiff(::TargetDiffusion) = false
```
indicating whether `σ(t, x, P::TargetDiffusion)` is independent from the values
of `x` and `t`,
```julia
clone(P::TargetDiffusion, θ) = TargetDiffusion(θ...)
```
which returns a new copy of the process with new set of parameters, and finally
```julia
params(P::TargetDiffusion) = [P.α, P.β]
```
which returns an array with all parameter values. Optionally, functions `nonhypo`, `hypo_a_inv`, `num_non_hypo` and `phi` can be defined to make it possible to perform conjugate updates of the parameters (see ... for more details on conjugate updates [TODO add])

## Definition of the auxiliary process
The auxiliary diffusion, similarly, needs to be a struct inheriting from
`ContinuousTimeProcess{ℝ{d,T}}` with the first type parameter `{T}` defining
the data-type of the parameters. It is often necessary for the auxiliary
diffusion to have to have access to the information regarding the starting
and ending time of the interval on which it is defined. Additionally, the
starting and end-point of the target process are also sometimes used (if
available). An example of a definition of the auxiliary diffusion is:
```julia
struct AuxiliaryDiffusion{R,S1,S2}
    α::R
    β::R
    t::Float64    # starting time of the interval
    u::S1         # starting position of the target process
    T::Float64    # end-time of the interval
    v::S2         # final position of the target process
    AuxiliaryDiffusion(α::R, β::R, t, u::S1, T, v::S2) = new{R,S1,S2}(α, β, t, u, T, v)
end
```
It is now necessary to specify the dynamics of the process. Unlike in the case
of the `Target`, specifying the volatility coefficient is not necessary and it
is sufficient to only provide a diffusion coefficient (`a:=σσ'`). The package
supports only linear diffusions as auxiliary processes and thus function `b`
should be defined as:
```julia
b(t, x, P::AuxiliaryDiffusion) = B(t, P) * x + β(t, P)
```
where `B` and `β` need to be overwritten as follows:
```julia
B(t, P::AuxiliaryDiffusion) = foo2(t, P)
β(t, P::AuxiliaryDiffusion) = foo3(t, P)
```
where the user-defined functions `foo2` and `foo3` should return a `d` by `d`
matrix and a length-`d` vector respectively. One needs to define the diffusion
coefficient
```julia
a(t, P::AuxiliaryDiffusion) = bar2(t, P)
```
As previously, the `clone` constructor, `params` and also a convenience
function returning the names of the paramters:
```julia
clone(P::AuxiliaryDiffusion, θ) = AuxiliaryDiffusion(θ..., P.t, P.u, P.T, P.v)
params(P::AuxiliaryDiffusion) = (P.α, P.β)
param_names(P::AuxiliaryDiffusion) = (:α, :β)
```
