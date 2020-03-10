using StaticArrays
"""
    readDataJRmodel(::Val{false}, filename; x0 = NaN)
read partial observations of the Jansen and Rit model according
to observation scheme given in https://arxiv.org/abs/1903.01138
"""
function readData_ecofin_model(filename1, filename2)
    df1 = CSV.read(filename1)
    df2 = CSV.read(filename2)
    obs = Vector{Any}()
    obsTime = Vector{Float64}()
    push!(obs, ℝ{3}(df2.x1[1], df2.x2[1], df1.x3[1]))
    j = 1
    for i in 2:length(df1.time)
        if (i-1)%90 == 0.0
            j += 1
            push!(obs, ℝ{3}(df2.x1[j], df2.x2[j], df1.x3[i]))
        else
            push!(obs,  ℝ{1}(df1.x3[i]))
        end
    end
    obsTime = df1.time
    x0 = ℝ{3}(df2.x1[1], df2.x2[1], df1.x3[1])
    fpt = [NaN for _ in obsTime[2:end]]
    fptOrPartObs = PartObs()
    df1, df2, x0, obs, obsTime, fpt, fptOrPartObs
end


filename1 = "./output/ecofin_path_interest_rates.csv"
filename2 = "./output/ecofin_path_indexes.csv"
df1, df2, x0, obs, obsTime, fpt, fptOrPartObs = readData_ecofin_model(filename1, filename2)
