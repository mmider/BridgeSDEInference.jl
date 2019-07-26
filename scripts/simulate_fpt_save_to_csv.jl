using Bridge
 using Random
 using DataFrames
 using CSV

 SRC_DIR = joinpath(Base.source_dir(), "..", "src")
 AUX_DIR = joinpath(SRC_DIR, "auxiliary")
 include(joinpath(AUX_DIR, "data_simulation_fns.jl"))
 OUT_DIR = joinpath(Base.source_dir(), "..", "output")
 mkpath(OUT_DIR)


 include(joinpath(SRC_DIR, "fitzHughNagumo.jl"))

 param = :simpleConjug
 FILENAME_OUT = joinpath(OUT_DIR,
                         "test_path_fpt_"*String(param)*".csv")
 let
     P = FitzhughDiffusion(param, 10.0, -8.0, 25.0, 0.0, 3.0)
     # starting point under :regular parametrisation
     x0 = ℝ{2}(-0.5, 0.6)
     # tranlate to conjugate parametrisation
     x0 = regularToConjug(x0, P.ϵ, 0.0)

     dt = 1/50000
     T = 10.0
     tt = 0.0:dt:T

     # due to memory constraints simulation needs to be done iteratively on segments

     # start from indicator that the down-crossing has already occured
     recentlyUpSearch = true
     tt_temp = copy(tt)
     Random.seed!(4)

     upLvl = 0.5
     downLvl = -0.5
     # Effective simulation of FPT over N*T interval
     N = 4
     upCrossingTimes = Float64[]
     upCrossingLvls = Float64[]
     downCrossingLvls = Float64[]
     x2 = Float64[]
     x0_perm = copy(x0)

     for i in 1:N
         # simulate path segment
         XX, _ = simulateSegment(0.0, x0, P, tt_temp)
         # determine all relevant up-crossings on it
         upCrossings, recentlyUpSearch = findCrossings(XX, upLvl, downLvl,
                                                       recentlyUpSearch)

         upCrossingTimes = vcat(upCrossingTimes, upCrossings)
         upCrossingLvls = vcat(upCrossingLvls, fill(upLvl, length(upCrossings)))
         downCrossingLvls = vcat(downCrossingLvls, fill(downLvl, length(upCrossings)))
         x2 = vcat(x2, fill(NaN, length(upCrossings)))

         x0 = XX.yy[end]
         tt_temp = tt .+ tt_temp[end]
     end

     # append info about the starting point:
     upCrossingTimes = vcat(0.0, upCrossingTimes)
     upCrossingLvls = vcat(x0_perm[1], upCrossingLvls)
     downCrossingLvls = vcat(downLvl, downCrossingLvls)
     x2 = vcat(x0_perm[2], x2)

     df = DataFrame(time=upCrossingTimes,
                    upCross=upCrossingLvls,
                    downCross=downCrossingLvls,
                    x2=x2)

     CSV.write(FILENAME_OUT, df)
 end
