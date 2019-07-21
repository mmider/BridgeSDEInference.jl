"""
    transformMCMCOutput(x0, paths, saveIter; chain=nothing, θ=nothing,
                        numGibbsSteps::Integer=1, parametrisation=:regular)

Transform the output of an MCMC inference procedure to be consistent with the
plotting functions
...
# Arguments
- `x0`: starting point under paramatrisation in which inference was performed
- `paths`: vector of saved paths
- `saveIter`: every one in every `saveIter` steps of MCMC chain saves a path
- `chain`: MCMC chain of parameters (if nothing, θ is used for all paths)
- `θ`: parameter (if nothing, θs from `chain` are used)
- `numGibbbsSteps`: number of steps in a single Gibbs sweep
- `parametrisation`: parametrisation of FitzHugh-Nagumo used during inference
...
"""
function transformMCMCOutput(x0, paths, saveIter; chain=nothing, θ=nothing,
                             numGibbsSteps::Integer=1, parametrisation=:regular,
                             warmUp=0)
    skip = numGibbsSteps * saveIter

    if chain == nothing
        @assert θ != nothing
        θs = [θ for i in 1:length(paths)]
    else
        @assert chain != nothing
        θs = chain[[i-warmUp for i in warmUp+1:warmUp+length(chain)
                    if i % saveIter == 0]]
    end

    if parametrisation == :regular
        pathsToSave = paths
        x0_new = x0
    elseif parametrisation in (:simpleAlter, :complexAlter)
        pathsToSave = [[alterToRegular(e, ϑ[1], ϑ[2]) for e in path]
                       for (path, ϑ) in zip(paths, θs)]
        x0_new = alterToRegular(x0, θs[1][1], θs[1][2])
    elseif parametrisation in (:simpleConjug, :complexConjug)
        pathsToSave = [[conjugToRegular(e, ϑ[1], 0) for e in path]
                        for (path, ϑ) in zip(paths, θs)]
        x0_new =  conjugToRegular(x0, θs[1][1], 0)
    else
        print("Unknown parametrisation\n")
    end
    x0_new, pathsToSave
end
