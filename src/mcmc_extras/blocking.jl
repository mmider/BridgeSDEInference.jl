import Base: display, length

struct ChequeredBlocking <: BlockingSchedule end


struct Block{S1,S2,S3,S4}
    Ls::Vector{S1}         # observation operators for both sets of blocks
    vs::Vector{S2}         # copied over observations of the process
    Σs::Vector{S3}         # covariance matrix of the noise for both sets of blocks
    aux_flags::Vector{S4}
    # two sets of knots
    knots::Vector{Int64}
    # two sets of blocks, where each block are indicies of intervals that make up a block
    blocks::Vector{Vector{Int64}}
    # info about the points at which to switch between the systems of ODEs
    change_pts::Vector{ODEChangePt}


    function Block(Ls::Vector{S1}, vs::Vector{S2}, Σs::Vector{S3},
                   aux_flags::Vector{S4}, knots::Vector{Int64},
                   blocks::Vector{Vector{Int64}},
                   change_pts::Vector{ODEChangePt}) where {S1,S2,S3,S4}
        new{S1,S2,S3,S4}(Ls, vs, Σs, aux_flags, knots, blocks, change_pts)
    end
end

length(bl::Block) = length(bl.blocks)
length(::NoBlocking) = 1

"""
    find_end_pts(𝔅::ChequeredBlocking, XX, idx)

Determine the observations for the update of the `idx`-th set of blocks. In
particular, on each block with interval indices [a₁,…,aₙ], observations vᵢ with
i∈{1,…,n-1} are made, whereas the last obesrvation is an exactly observed
process Xₜ at the terminal time t of the aₙ-th interval.
"""
function find_end_pts(bl::Block, XX, idx)
    [( k in bl.knots[idx] ? X.yy[end] : bl.vs[k]) for (k,X) in enumerate(XX)]
end


"""
    display(𝔅::NoBlocking)

Nothing particular to display
"""
function display(bl::NoBlocking)
    print("No blocking...\n")
end

"""
    display(𝔅::ChequeredBlocking)

Display the pattern of blocks
"""
function display(bl::Block)
    function print_blocks(knots, idxLast, m)
        M = length(knots)
        get_knot(knots, i) = (M >= i > 0 ? knots[i] : idxLast * (i>0))
        function print_range(from, to)
            for i in from:to
                print("|", get_knot(knots,i-1), "|----",
                      get_knot(knots,i)-get_knot(knots,i-1), "----")
            end
        end

        if m > div(M, 2)
            print_range(1, M+1)
            print("|", idxLast, "|")
        else
            print_range(1,m)
            print("|", get_knot(knots, m) ,"|   ...   ")
            print_range(M+2-m,M+1)
            print("|", idxLast, "|")
        end
        print("  (number of blocks: ", M+1,")")
    end
    print("Chequered Blocking scheme\n",
          "-------------------------\n",
          "Format:\n",
          "block sizes in A: ")
    print_blocks(𝔅.knots[1], length(𝔅.vs), 3)
    print("\nblock sizes in B: ")
    print_blocks(𝔅.knots[2], length(𝔅.vs), 3)
    print("\n")
end




function _print_info(info)
    m = length(info)
    for i in 1:m
        print("b", i, ": ", round(info[i], digits=3), " | ")
    end
    print("\n- - - - - - - - - - - - - -\n")
end



function create_blocks(::ChequeredBlocking, P, params)
    knots, ϵ, change_pt = params
    find_knots(mod, rem) = [k for (i,k) in enumerate(knots) if i % mod == rem]
    knotsA, knotsB = find_knots(2, 1), find_knots(2, 0)

    m, d = size(P[end].L)

    find_L(knots) = [( k in knots ? SMatrix{d,d}(1.0*I) : p.L)
                     for (k,p) in enumerate(P)]
    LsA, LsB = find_L(knotsA), find_L(knotsB)

    vs = [p.v for p in P]

    find_Σ(knots) = [(k in knots ? SMatrix{d,d}(ϵ*I) : p.Σ)
                     for (k,p) in enumerate(P)]
    ΣsA, ΣsB = find_Σ(knotsA), find_Σ(knotsB)

    find_ch_pt(knots) = [(k in knots ? deepcopy(change_pt) : p.change_pt)
                         for (k,p) in enumerate(P)]
    chpA, chpB = find_ch_pt(knotsA), find_ch_pt(knotsB)

    find_aux_flag(knots) = [( k in knots ? Nothing : get_aux_flag(p.Pt))
                            for (k,p) in enumerate(P)]
    auxA, auxB = find_aux_flag(knotsA), find_aux_flag(knotsB)

    """
        knots_to_blocks(knots, idxLast, i)

    Given a list of `knots` fetch the indices of the intervals that together
    make up the `i`-th block. `idxLast` is the index of the last interval.
    """
    function knots_to_blocks(knots, idx_last, i)
        M = length(knots)
        @assert M > 0
        if M >= i > 1
            return (knots[i-1]+1):knots[i]
        elseif i == 1
            return 1:knots[1]
        else
            return (knots[M]+1):idx_last
        end
    end
    blocks = ([collect(knots_to_blocks(knotsA, length(P), i)) for i in 1:length(knotsA)+1],
              [collect(knots_to_blocks(knotsB, length(P), i)) for i in 1:length(knotsB)+1])

    (Block(LsA, vs, ΣsA, auxA, knotsA, blocks[1], chpA),
     Block(LsB, vs, ΣsB, auxB, knotsB, blocks[2], chpB))
end

create_blocks(blocking::NoBlocking, P, params) = blocking
