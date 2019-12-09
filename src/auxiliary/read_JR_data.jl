"""
    readDataJRmodel(::Val{false}, filename; x0 = NaN)
read partial observations of the Jansen and Rit model according
to observation scheme given in https://arxiv.org/abs/1903.01138
"""
function readDataJRmodel(::Val{false}, filename; x0 = NaN)
    df = CSV.read(filename)
    obs = ℝ{1}.(df.x1)
    obsTime = Float64.(df.time)
    if isnan(NaN)
        x0 = ℝ{1}(df.x1[1])
    end
    fpt = [NaN for _ in obsTime[2:end]]
    fptOrPartObs = PartObs()
    df, x0, obs, obsTime, fpt, fptOrPartObs
end
