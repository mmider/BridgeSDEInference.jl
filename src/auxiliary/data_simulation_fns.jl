"""
    simulateSegment(::S, x0, P, tt)

Simulate path of a target process `P`, started from `x0` on a time-grid `tt`
"""
function simulateSegment(::S, x0, P, tt) where S
    Wnr = Wiener{S}()
    WW = Bridge.samplepath(tt, zero(S))
    sample!(WW, Wnr)
    # forcedSolve will allow sampling of domain-restricted diffusions
    WW, XX = forcedSolve(EulerMaruyamaBounded(), x0, WW, P)
    XX, XX.yy[end]
end

"""
    findCrossings(XX, upLvl, downLvl, upSearch=false)

Find times of first up-crossings of level `upLvl`. The timing starts only after
the diffusion was brought below level `downLvl` first (or alternatively if
`upSearch` flag is set to true)
"""
function findCrossings(XX, upLvl, downLvl, upSearch=false)
    upSearch=upSearch
    upCrossingTimes = Float64[]
    for (x,t) in zip(XX.yy, XX.tt)
        if upSearch && x[1] > upLvl
            upSearch = false
            push!(upCrossingTimes, t)
        elseif !upSearch && x[1] < downLvl
            upSearch = true
        end
    end
    upCrossingTimes, upSearch
end
