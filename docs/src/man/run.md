# Running the MCMC sampler
Running the MCMC sampler is very simple, it is enough to call
```@docs
mcmc
```
passing the initialised `MCMCSetup`, `MCMCSchedule` and `DiffusionSetup`
objects, as follows:
```julia
out = mcmc(mcmc_setup, schedule, model_setup)
```
