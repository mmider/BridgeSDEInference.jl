# Querying the inference results
The output of running the `mcmc` function is a `Workspace` object.
```@docs
Workspace
```
Querying the output can be done simply by calling the members of an instance of
`Workspace` that is returned by the `mcmc` sampler. The task of writing
suitable functions for this is left mainly to the user, we provide some generic
plotting functions that can be used for testing
[here](https://github.com/mmider/BridgeSDEInference.jl/blob/master/auxiliary/plotting_fns.jl) (not part of the package).
