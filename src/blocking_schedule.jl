import Base: display

"""
    BlockingSchedule

Types inheriting from abstract type `BlockingSchedule` define the schedule
according to which blocking is done
"""
abstract type BlockingSchedule end


"""
    struct NoBlocking <: BlockingSchedule end

Regular updates with no blocking
"""
struct NoBlocking <: BlockingSchedule end


"""
    ChequeredBlocking

Struct
```
struct ChequeredBlocking{TP,TWW,TXX} <: BlockingSchedule
    P::TP      # blocking workspace: diffusion law
    WW::TWW    # blocking workspace: accepted Wiener path
    WWᵒ::TWW   # blocking workspace: proposed Wiener path
    XX::TXX    # blocking workspace: accepted diffusion path
    XXᵒ::TXX   # blocking workspace: proposed diffusion path
    Ls         # observation operators for both sets of blocks
    vs         # copied over observations of the process
    Σs         # covariance matrix of the noise for both sets of blocks
    # two sets of knots
    knots::Tuple{Vector{Int64}, Vector{Int64}}
    # two sets of blocks, where each block are indicies of intervals that make up a block
    blocks::Tuple{Vector{Vector{Int64}}, Vector{Vector{Int64}}}
    idx::Int64 # index of set of blocks that are being updated ∈{1,2}
    accpt::Tuple{Vector{Int64}, Vector{Int64}} # tracker for the number of accepted samples
    props::Tuple{Vector{Int64}, Vector{Int64}} # tracker for the number of proposed samples
    # info about the points at which to switch between the systems of ODEs
    changePts::Tuple{Vector{ODEChangePt}, Vector{ODEChangePt}}
end
```
is a blocking schedule in which two sets of blocks are defined in an interlacing
manner. For instance, if all knots consist of {1,2,3,4,5,6,7,8,9}, then set A
will contain {1,3,5,7,9}, whereas set B {2,4,6,8}. These knots will then
uniquely determine the blocks

    ChequeredBlocking(knots::Vector{Int64}, ϵ::Float64, changePt::ODEChangePt,
                      P::TP, WW::TWW, XX::TXX)

Base constructor that takes a set of all `knots` (which it then splits into
two disjoint subsets), the artificial noise parameter `ϵ`, information about
a point at which to switch between ODE solvers for H, Hν, c and L, M⁺, μ, an
object with diffusion laws `P`, container `WW` for the driving Brownian motion,
container `XX` for the sampled path (only the size and type of the latter three
are important, not actual values passed, because they are copied and used
internally as containers).

    ChequeredBlocking(𝔅::ChequeredBlocking{TP̃, TWW, TXX}, P::TP, idx::Int64)

Clone constructor. It creates a new object `ChequeredBlocking` from the old one
`𝔅` by using the same type and size of the containers `WW` and `XX`, but a
changed type of the law `P` (changed from `TP̃` to `TP`). The index of a current
update can also be changed via `idx`.

    ChequeredBlocking()

Empty constructor.
"""
struct ChequeredBlocking{TP,TWW,TXX} <: BlockingSchedule
    P::TP      # blocking workspace: diffusion law
    WW::TWW    # blocking workspace: accepted Wiener path
    WWᵒ::TWW   # blocking workspace: proposed Wiener path
    XX::TXX    # blocking workspace: accepted diffusion path
    XXᵒ::TXX   # blocking workspace: proposed diffusion path
    Ls         # observation operators for both sets of blocks
    vs         # copied over observations of the process
    Σs         # covariance matrix of the noise for both sets of blocks
    # two sets of knots
    knots::Tuple{Vector{Int64}, Vector{Int64}}
    # two sets of blocks, where each block are indicies of intervals that make up a block
    blocks::Tuple{Vector{Vector{Int64}}, Vector{Vector{Int64}}}
    idx::Int64 # index of set of blocks that are being updated ∈{1,2}
    accpt::Tuple{Vector{Int64}, Vector{Int64}} # tracker for the number of accepted samples
    props::Tuple{Vector{Int64}, Vector{Int64}} # tracker for the number of proposed samples
    # info about the points at which to switch between the systems of ODEs
    changePts::Tuple{Vector{ODEChangePt}, Vector{ODEChangePt}}

    function ChequeredBlocking(knots::Vector{Int64}, ϵ::Float64,
                               changePt::ODEChangePt, P::TP, WW::TWW, XX::TXX
                               ) where {TP,TWW,TXX}
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

        findChP(knots) = [(k in knots ? deepcopy(changePt) : p.changePt)
                                                    for (k,p) in enumerate(P)]
        chpA = findChP(knotsA)
        chpB = findChP(knotsB)

        """
            knotsToBlocks(knots, idxLast, i)

        Given a list of `knots` fetch the indices of the intervals that together
        make up the `i`-th block. `idxLast` is the index of the last interval.
        """
        function knotsToBlocks(knots, idxLast, i)
            M = length(knots)
            if M >= i > 1
                return (knots[i-1]+1):knots[i]
            elseif i == 1
                return 1:knots[1]
            else
                return (knots[M]+1):idxLast
            end
        end
        blocks = ([collect(knotsToBlocks(knotsA, length(P), i)) for i in 1:length(knotsA)+1],
                  [collect(knotsToBlocks(knotsB, length(P), i)) for i in 1:length(knotsB)+1])

        accpt = (zeros(Int64, length(blocks[1])),
                 zeros(Int64, length(blocks[2])))
        props = (zeros(Int64, length(blocks[1])),
                 zeros(Int64, length(blocks[2])))
        new{TP,TWW,TXX}(deepcopy(P), deepcopy(WW), deepcopy(WW), deepcopy(XX),
                        deepcopy(XX), (LsA, LsB), vs, (ΣsA, ΣsB),
                        (knotsA, knotsB), blocks, 1, accpt, props,
                        (chpA, chpB))
    end

    function ChequeredBlocking(𝔅::ChequeredBlocking{TP̃, TWW, TXX}, P::TP,
                               idx::Int64) where {TP̃,TP,TWW,TXX}
        new{TP,TWW,TXX}(P, 𝔅.WW, 𝔅.WWᵒ, 𝔅.XX, 𝔅.XXᵒ, 𝔅.Ls, 𝔅.vs, 𝔅.Σs,
                        𝔅.knots, 𝔅.blocks, idx, 𝔅.accpt, 𝔅.props, 𝔅.changePts)
    end

    function ChequeredBlocking()
        new{Nothing, Nothing, Nothing}(nothing, nothing, nothing, nothing,
                                       nothing, nothing, nothing, nothing,
                                       ([0],[0]),([[0]],[[0]]), 1, ([0],[0]),
                                       ([0],[0]),
                                       ([NoChangePt()],[NoChangePt()])
                                       )
    end
end


"""
    findEndPts(𝔅::ChequeredBlocking, XX, idx)

Determine the observations for the update of the `idx`-th set of blocks. In
particular, on each block with interval indices [a₁,…,aₙ], observations vᵢ with
i∈{1,…,n-1} are made, whereas the last obesrvation is an exactly observed
process Xₜ at the terminal time t of the aₙ-th interval.
"""
function findEndPts(𝔅::ChequeredBlocking, XX, idx)
    [( k in 𝔅.knots[idx] ? X.yy[end] : 𝔅.vs[k]) for (k,X) in enumerate(XX)]
end

"""
    next(𝔅::ChequeredBlocking, XX, θ)

Switch the set of blocks that are being updated. `XX` is the most recently
sampled (accepted) path. `θ` can be used to change parametrisation.
"""
function next(𝔅::ChequeredBlocking, XX, θ)
    newIdx = (𝔅.idx % 2) + 1
    vs = findEndPts(𝔅, XX, newIdx)
    Ls = 𝔅.Ls[newIdx]
    Σs = 𝔅.Σs[newIdx]
    chPts = 𝔅.changePts[newIdx]

    P = [GuidPropBridge(𝔅.P[i], Ls[i], vs[i], Σs[i], chPts[i], θ)
                                            for (i,_) in enumerate(𝔅.P)]

    ChequeredBlocking(𝔅, P, newIdx)
end

"""
    display(𝔅::NoBlocking)

Nothing particular to display
"""
function display(𝔅::NoBlocking)
    print("No blocking...\n")
end

"""
    display(𝔅::ChequeredBlocking)

Display the pattern of blocks
"""
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

"""
    registerAccpt(𝔅::BlockingSchedule, i, accepted)

Register whether the block has been accepted
"""
function registerAccpt!(𝔅::BlockingSchedule, i, accepted)
    𝔅.props[𝔅.idx][i] += 1
    𝔅.accpt[𝔅.idx][i] += 1*accepted
end


"""
    displayAcceptance(𝔅::NoBlocking)

Nothing to display
"""
function displayAcceptanceRate(𝔅::NoBlocking) end


"""
    displayAcceptance(𝔅::NoBlocking)

Display acceptance rates
"""
function displayAcceptanceRate(𝔅::BlockingSchedule)
    print("\nAcceptance rates:\n----------------------\n")
    function printAccptRate(accpt, prop)
        m = length(accpt)
        for i in 1:m
            print("b", i, ": ", accpt[i]/prop[i], " | ")
        end
        print("\n- - - - - - - - - - - - - -\n")
    end
    printAccptRate(𝔅.accpt[1], 𝔅.props[1])
    printAccptRate(𝔅.accpt[2], 𝔅.props[2])
end
