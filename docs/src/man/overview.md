# Overview of the workflow

A typical workflow consists of five stages (or six if the data also needs to be
generated).

## Stage 1 (a)
Define a diffusion model. Some examples are included in the folder
[src/examples](https://github.com/mmider/BridgeSDEInference.jl/blob/master/src/examples/).
See [Definition of a diffusion process](@ref) for a list of functions that
need (and optionally might) be implemented to fully define a diffusion process.

## Stage 1 (b)
Generate the data. Naturally in applied work the data is given and this step is
skipped. However, for tests it is often convenient to work with simulated data.
Data generation is not an internal part of the package; however some generic
methods that allow for simulation of observations can be found in folders (...).
See also [Data Generation](./generate_data.md).

## Stage 2
Define the model `setup`. This includes the type of a model together with
all the necessary parametrisations is needed to fully spedcfor  of the

## Stage 3
Define the MCMC chain. This amounts to specifying the range of possible
transition steps (together with their transition kernels and priors) and the
MCMC schedule which lists the order of steps and actions that need to be
undertaken by the MCMC sampler.

## Stage 4
Run the `mcmc` function. This is a one-liner:
```julia
out = mcmc(mcmc_setup, schedule, model_setup)
```
where `mcmc_setup`, `schedule`, `model_setup` are defined in the stages 2 &
3 above.

## Stage 5
Query the results. This step is mostly left to a user. There are a couple of
generic plotting functions defined in (...), which are not part of the package.
The output are two objects: `Workspace` and `MCMCWorkspace`, please see
[Querying the inference results](@ref) for an overview of their members and
some auxiliary functions that can be used to visualise the results.
