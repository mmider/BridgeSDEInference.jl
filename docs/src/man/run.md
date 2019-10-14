# Running the MCMC sampler
Running the MCMC sampler is very simple, it is enough to call
```@docs
mcmc
```
passing the initialised `MCMCsetup` object, as follows:
```julia
out = mcmc(setup)
```
