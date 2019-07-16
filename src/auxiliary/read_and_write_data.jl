using DataFrames
using CSV

"""
    readData(::Val{true}, filename)

Reads first passage time data from a file `filename` (assumed to be saved in a
correct format---no checks are performed).
"""
function readData(::Val{true}, filename)
    df = CSV.read(filename)
    x0 = ℝ{2}(df.upCross[1], df.x2[1])
    obs = ℝ{1}.(df.upCross)
    obsTime = Float64.(df.time)
    fpt = [FPTInfo((1,), (true,), (resetLvl,), (i==1,)) for
            (i, resetLvl) in enumerate(df.downCross[2:end])]
    fptOrPartObs = FPT()
    df, x0, obs, obsTime, fpt, fptOrPartObs
end


"""
    readData(::Val{false}, filename)

Reads partial observation type data from a file `filename` (assumed to be saved
in a correct format---no checks are performed)
"""
function readData(::Val{false}, filename)
    df = CSV.read(filename)
    obs = ℝ{1}.(df.x1)
    obsTime = Float64.(df.time)
    x0 = ℝ{2}(df.x1[1], df.x2[1])
    fpt = [NaN for _ in obsTime[2:end]]
    fptOrPartObs = PartObs()
    df, x0, obs, obsTime, fpt, fptOrPartObs
end


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
