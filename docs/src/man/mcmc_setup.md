# Definition of the MCMC sampler
To define the MCMC sampler there are two objects that need to be defined. First,
we must specify a range of possible transition kernels that the MCMC sampler
can use (these transition kernels can change and adapt as the MCMC sampler
progresses if certain settings are turned on). These will be stored in the
object `MCMCSetup`. We then have to specify the schedule i.e. what steps need
to be taken by the MCMC sampler at each of its step.

## MCMC Setup
As in the previous section, we define transition steps for the problem of
inference for diffusion processes. We will need two types of updates:
imputations and parameter updates. Suppose that we don't use any blocking, to
see how to modify the code below to include blocking see
[this page](@ref blocking_header). Then imputation can be defined via:
```julia
precond_Crank_Nicolson = ...
ODE_solver = ... # possible choices Ralston3, RK4, Tsit5, Vern7
impute_step = Imputation(NoBlocking(), precond_Crank_Nicolson, ODE_solver)
```
```@docs
Imputation
```
To define parameter updates more parameters need to be defined:
```julia
pu1 = ParamUpdate(MetropolisHastingsUpdt(), # the type of parameter update
                  5,                        # which coordinate is updated
                  θ_init,                   # needed just for the dimension of the parameter
                  UniformRandomWalk(0.5, true), # transition kernel
                  ImproperPosPrior(),       # prior
                  UpdtAuxiliary(            # auxiliary information
                      Vern7(),              # ODE solver
                      true))                # whether the update prompts for re-computing H,Hnu,c
pu2 = ...
```
Then the setup can be defined as:
```julia
mcmc_setup = MCMCSetup(impute_step, pu1, pu2)
```
There are currently two different ways of updating parameters:
```@docs
ConjugateUpdt
MetropolisHastingsUpdt
```
And two generic transition kernels:
```@docs
UniformRandomWalk
GaussianRandomWalk
```

## MCMC schedule
It is now necessary to tell the mcmc sampler what to do at each iteration. To
this end let's define the `MCMCSchedule`
```@docs
MCMCSchedule
```
An example of how to define it is given here:
```julia
MCMCSchedule(111, [[1,2], [1,3]],
              (save=5, verbose=10, warm_up=7,
                readjust=(x->x%20==0), fuse=(x->false)))
```
It specifies the total number of MCMC steps to be `111`, and the array
`[[1,2], [1,3]]` makes the sampler alternate between performing imputation and
parameter update `pu1` on each odd step and imputation and parameter update
`pu2` on each even step. It saves the imputed path on every `5`th iteration,
prints out useful progress message to the console every `10`th step, ignores
parameter update steps for the first `7` iterations readjusts the proposals
once in every `20`th step (see [this page](@ref adaptive_header) for more
information on adaptive schemes). It also never fuses transition kernels (see
[this page](@ref fusion_header) for more information about fusion).
