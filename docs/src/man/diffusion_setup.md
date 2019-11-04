# Defining the observational scheme

To run the MCMC sampler we need to pass a `setup` object that defines the
statistical model. In this section we will show how to define a diffusion model
using a pre-defined `DiffusionSetup`. Nonetheless, be aware that the logic of
the MCMC sampler is independent from the model specification, so it is possible
to use the `mcmc` routines from this package routine to run it on other
statistical models. For more information about these types of extensions see
[Generic MCMC](@ref).

## Defining the processes
To define the `DiffusionSetup` we need to decide on the `Target` diffusion, an
`Auxiliary` diffusion and the observation scheme. For instance, suppose that
there are three observations:
```julia
obs = [...]
obs_times = [1.0, 2.0, 3.0]
```
Then we define the `target` diffusion globally and the `auxiliary` diffusion
separately on each interval
```julia
P_target = TargetDiffusion(parameters)
P_auxiliary = [AuxiliaryDiffusion(parameters, o, t) for (o,t) in zip(obs, obs_times)]
```
To define the setup for partially observed diffusion it is enough to write:
```julia
model_setup = DiffusionSetup(P_target, P_auxiliary, PartObs()) # for first passage times use FPT()
```
```@docs
DiffusionSetup
```

## Observations
To set the observations, apart from passing the observations and observation
times it is necessary to pass the observational operators and as well as
covariance of the noise. Additionally, one can pass additional information
about the [first passage time scheme](@ref FPT_header).
```julia
L = ...
Σ = ...
set_observations!(model_setup, [L for _ in obs], [Σ for _ in obs], obs, obs_time)
```
```@docs
set_observations!
```

## Imputation grid
There are two objects that define the imputation grid. The time step `dt` and
the time transformation that transforms a regular time-grid with equidistantly
distributed imputation points. The second defaults to a usual transformation
employed in papers on the guided proposals. [TO DO add also space-time
transformation from the original paper for the bridges]. It is enough to call
```julia
dt = ...
set_imputation_grid!(model_setup, dt)
```
```@docs
set_imputation_grid!
```



## Prior over the starting point
There are two types of priors for the starting point, either a `delta hit` at
some specified value, corresponding to a known starting point and a Gaussian
prior
```@docs
KnownStartingPt
GsnStartingPt
```
For instance, to set a known starting point it is enough to call:
```julia
set_x0_prior!(model_setup, KnownStartingPt(x0))
```
```@docs
set_x0_prior!
```



## Auxiliary parameters
There are two auxiliary parameters that can be set by the user. The first one
is a thinning factor for saving the paths that are sampled by the MCMC sampler.
The second specifies an adaptation scheme for tuning Guided proposals [TODO
change the latter].
```@docs
set_auxiliary!
```


## Initialisation of internal containers
Once all the setting functions above have been run (with the only exception
`set_auxiliary!` being optional), i.e.
```julia
setup = DiffusionSetup(...)
set_observations!(setup, ...)
set_imputation_grid(setup, ...)
set_x0_prior!(setup, ...)
```
then the following function should be run
```@docs
initialise!
```
Once run, the model setup is complete.
