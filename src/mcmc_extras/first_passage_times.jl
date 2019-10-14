"""
    FPTInfo{S,T}

The struct
```
struct FPTInfo{S,T}
    condCoord::NTuple{N,S}
    upCrossing::NTuple{N,Bool}
    autoRenewed::NTuple{N,Bool}
    reset::NTuple{N,T}
end
```
serves as a container for the information regarding first passage time
observations. `condCoord` is an NTuple of coordinates that are conditioned on
the first passage time nature of the observations. `upCrossing` indicates
whether observations of the corresponding coordinate are up-crossings or
down-crossings. `autoRenewed` indicates whether process starts from the
renewed state (i.e. normally the process is unconstrained until it hits level
`reset` for the first time, however `autoRenewed` process is constrained on the
first passage time from the very beginnig). `reset` level is the level that
needs to be reached before the process starts to be conditioned on the first
passage time.
"""
struct FPTInfo{S,T,N}
    condCoord::NTuple{N,S}
    upCrossing::NTuple{N,Bool}
    autoRenewed::NTuple{N,Bool}
    reset::NTuple{N,T}

    FPTInfo(condCoord::NTuple{N,S}, upCrossing::NTuple{N,Bool},
            reset::NTuple{N,T},
            autoRenewed::NTuple{N,Bool} = Tuple(fill(false,length(condCoord)))
            ) where {S,T,N} = new{S,T,N}(condCoord, upCrossing,
                                         autoRenewed, reset)
end


"""
    checkSingleCoordFpt(XXᵒ, c, cidx, fpt)

Verify whether coordinate `c` (with index number `cidx`) of path `XXᵒ`.yy
adheres to the first passage time observation scheme specified by the object
`fpt`.
"""
function checkSingleCoordFpt(XXᵒ, c, cidx, fpt)
    k = length(XXᵒ.yy)
    thrsd = XXᵒ.yy[end][c]
    renewed = fpt.autoRenewed[cidx]
    s = fpt.upCrossing[cidx] ? 1 : -1
    for i in 1:k
        if !renewed && (s*XXᵒ.yy[i][c] <= s*fpt.reset[cidx])
            renewed = true
        elseif renewed && (s*XXᵒ.yy[i][c] > s*thrsd)
            return false
        end
    end
    return renewed
end


"""
    checkFpt(::PartObs, XXᵒ, fpt)

First passage time constrains are automatically satisfied for the partially
observed scheme
"""
checkFpt(::PartObs, XXᵒ, fpt) = true


"""
    checkFpt(::FPT, XXᵒ, fpt)

Verify whether path `XXᵒ`.yy adheres to the first passage time observation
scheme specified by the object `fpt`.
"""
function checkFpt(::FPT, XXᵒ, fpt)
    for (cidx, c) in enumerate(fpt.condCoord)
        if !checkSingleCoordFpt(XXᵒ, c, cidx, fpt)
            return false
        end
    end
    return true
end


"""
    check_full_path_fpt(::PartObs, ::Any, ::Any, ::Any)

First passage time constrains are automatically satisfied for the partially
observed scheme
"""
check_full_path_fpt(::PartObs, ::Any, ::Any, ::Any) = true


"""
    check_full_path_fpt(::PartObs, XXᵒ, m, fpt)

Verify whether all paths in the range `iRange`, i.e. `XXᵒ`[i].yy, i in `iRange`
adhere to the first passage time observation scheme specified by the object
`fpt`
"""
function check_full_path_fpt(::FPT, XXᵒ, iRange, fpt)
    for i in iRange
        if !checkFpt(FPT(), XXᵒ[i], fpt[i])
            return false
        end
    end
    return true
end
