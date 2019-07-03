


df2 = savePathsToFile(pathsToSave, time_, joinpath(outdir, "sampled_paths.csv"))
df3 = saveChainToFile(chain, joinpath(outdir, "chain.csv"))
```

Lastly, we can make some diagnostic plots:
```julia
include("src/plots.jl")
# make some plots
set_default_plot_size(30cm, 20cm)
if fptObsFlag
    plotPaths(df2, obs=[Float64.(df.upCross), [x0[2]]],
              obsTime=[Float64.(df.time), [0.0]],obsCoords=[1,2])
else
    plotPaths(df2, obs=[Float64.(df.x1), [x0[2]]],
              obsTime=[Float64.(df.time), [0.0]],obsCoords=[1,2])
end
plotChain(df3, coords=[1])
plotChain(df3, coords=[2])
plotChain(df3, coords=[3])
plotChain(df3, coords=[5])
```
Here are the results, the sampled paths:

![temp](../assets/paths.js.svg)

And the Markov chains, for parameter ϵ:

![temp](../assets/param1.js.svg)

parameter s:

![temp](../assets/param2.js.svg)

parameter γ:

![temp](../assets/param3.js.svg)

and parameter σ:

![temp](../assets/param5.js.svg)