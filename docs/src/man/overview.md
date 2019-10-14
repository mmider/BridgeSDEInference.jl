# Overview of the workflow

A typical workflow consists of four stages (or five if the data also needs to be
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
Define the observational `setup`, together with the parametrisation of the
Markov chain Monte Carlo sampler. There are a lot of elements that can be
tailored by the user at this step; however, much of the heavy work of code
writing is taken care of by the internal routines of the package and the amount
of code that needs to be written by the user is conveniently kept to a minimum.
This comes at a cost of having to familiarise oneself with the syntax used to
define the observational scheme and the MCMC; please see [Setup](@ref)
to view available options regarding the `setup`.

## Stage 3
Run the `mcmc` function. This is a one-liner:
```julia
out = mcmc(setup)
```
where `setup` is an appropriately defined observational setup and definition
of the Markov chain.

## Stage 4
Query the results. This step is mostly left to a user. There are a couple of
generic plotting functions defined in (...), which are not part of the package.
The output is an object `Workspace`, please see [Query](@ref) for an
overview of its members and some auxiliary functions that can be used to
visualise the results.
