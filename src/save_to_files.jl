using DataFrames
using CSV

"""
    savePathsToFile(paths, time, filename)

Put diffusion paths and respective times to a dataframe and save to a file
"""
function savePathsToFile(paths, time, filename)
    idx = collect( Iterators.flatten([fill(i,length(X))
                            for (i,X) in enumerate(paths)]))
    time = collect(Iterators.flatten(fill(time,length(paths))))
    paths = collect(Iterators.flatten(paths))
    numDims = length(paths[1])
    df = DataFrame(idx=idx, time=time)
    for i in 1:numDims
        X = [pt[i] for pt in paths]
        df = hcat(df, X, makeunique=true)
    end
    CSV.write(filename, df)
    df
end


"""
    saveChainToFile(chains, filename)

Save MCMC chain to a file
"""
function saveChainToFile(chains, filename)
    df = DataFrame()
    for i in 1:length(chains[1])
        X = [pt[i] for pt in chains]
        df = hcat(df, X, makeunique=true)
    end
    CSV.write(filename, df)
    df
end
