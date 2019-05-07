import Base: display

abstract type BlockingSchedule end

struct NoBlocking <: BlockingSchedule end


struct ChequeredBlocking{TP,TWW,TXX} <: BlockingSchedule
    P::TP
    WW::TWW
    WWᵒ::TWW
    XX::TXX
    XXᵒ::TXX
    Ls
    vs
    Σs
    knots::Tuple{Vector{Int64}, Vector{Int64}}
    idx::Int64

    function ChequeredBlocking(knots::Vector{Int64}, ϵ::Float64, P::TP, WW::TWW,
                               XX::TXX) where {TP,TWW,TXX}
        findKnots(mod, rem) = [k for (i,k) in enumerate(knots) if i % mod == rem]
        knotsA = findKnots(2, 1)
        knotsB = findKnots(2, 0)

        m, d = size(P[end].L)

        findL(knots) = [( k in knots ? SMatrix{d,d}(1.0*I) : p.L) for (k,p) in enumerate(P)]
        LsA = findL(knotsA)
        LsB = findL(knotsB)

        vs = [p.v for p in P]

        findΣ(knots) = [(k in knots ? SMatrix{d,d}(ϵ*I) : p.Σ) for (k,p) in enumerate(P)]
        ΣsA = findΣ(knotsA)
        ΣsB = findΣ(knotsB)

        new{TP,TWW,TXX}(deepcopy(P), deepcopy(WW), deepcopy(WW), deepcopy(XX),
                        deepcopy(XX), (LsA, LsB), vs, (ΣsA, ΣsB),
                        (knotsA, knotsB), 1)
    end

    function ChequeredBlocking(𝔅::ChequeredBlocking{TP̃, TWW, TXX}, P::TP,
                               idx::Int64) where {TP̃,TP,TWW,TXX}
        new{TP,TWW,TXX}(P, 𝔅.XX, 𝔅.XXᵒ, 𝔅.WW, 𝔅.WWᵒ, 𝔅.Ls, 𝔅.vs, 𝔅.Σs,
                        𝔅.knots, idx)
    end

    function ChequeredBlocking()
        new{Nothing, Nothing, Nothing}(nothing, nothing, nothing, nothing,
                                       nothing, nothing, nothing, nothing,
                                       ([0.],[0.]), 1)
    end
end

function findEndPts(𝔅::ChequeredBlocking, XX, idx)
    [( k in 𝔅.knots[idx] ? 𝔅.vs[k] : X[end]) for (k,X) in enumerate(XX)]
end

function next(𝔅::ChequeredBlocking, XX)
    newIdx = (𝔅.idx % 2) + 1
    vs = findEndPts(𝔅, XX, newIdx)
    Ls = 𝔅.Ls[newIdx]
    Σs = 𝔅.Σs[newIdx]

    P = [GuidPropBridge(𝔅.P[i], Ls[i], vs[i], Σs[i]) for (i,_) in enumerate(𝔅.P)]

    ChequeredBlocking(𝔅, P, newIdx)
end

function display(𝔅::NoBlocking)
    print("No blocking...\n")
end


function display(𝔅::ChequeredBlocking)
    function printBlocks(knots, idxLast, m)
        M = length(knots)
        getKnot(knots, i) = (M >= i > 0 ? knots[i] : idxLast * (i>0))
        function printRange(from, to)
            for i in from:to
                print("|", getKnot(knots,i-1), "|----",
                      getKnot(knots,i)-getKnot(knots,i-1), "----")
            end
        end

        if m > div(M, 2)
            printRange(1, M+1)
            print("|", idxLast, "|")
        else
            printRange(1,m)
            print("|", getKnot(knots, m) ,"|   ...   ")
            printRange(M+2-m,M+1)
            print("|", idxLast, "|")
        end
        print("  (number of blocks: ", M+1,")")
    end
    print("Chequered Blocking scheme\n",
          "-------------------------\n",
          "Format:\n",
          "block sizes in A: ")
    printBlocks(𝔅.knots[1], length(𝔅.P), 3)
    print("\nblock sizes in B: ")
    printBlocks(𝔅.knots[2], length(𝔅.P), 3)
    print("\n")
end
