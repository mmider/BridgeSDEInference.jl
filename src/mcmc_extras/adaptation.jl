#=
    --------------------------------------------------------------------------
    Implements functionality for learning a mean diffusion's trajectory during
    mcmc sampling. The main object is `Adaptation`.
    --------------------------------------------------------------------------
=#
import Base.resize!

"""
    Adaptation{TV,T}

Stores the history of imputed paths from which the mean trajectory can be
computed, as well as the scheme according to which the adaptation is supposed
to be performed. `TV` indicates whether any adaptation should be done at all.
`TV` set to `Val{False}` acts as an indicator that no adaptation is to be done.
"""
struct Adaptation{TV,T,S}
    X::Vector{Vector{Vector{T}}} # history of paths
    # NOTE `ρs` will be removed anyway after introducing adaptive proposals
    ρs::Vector{S}          # ladder of memory param for precond. Crank-Nicolson
    λs::Vector{Float64}          # ladder of weights that balance initial
                                 # auxiliary law and the adaptive law based on
                                 # the mean trajectory
    sizes::Vector{Int64}         # ladder indicating number of paths that are to
                                 # be used for computing mean trajectory
    skip::Int64                  # save 1 in every ... many sampled paths
    N::Vector{Int64}             # counter #1-current position on the ladder
                                 # #2-current index of the last saved path

    """
        Adaptation(::T, ρs, λs, sizes_of_path_coll, skip=1)

    Initialise adaptation. `ρs`, `λs` and `sizes_of_path_coll` are ladders that
    are traversed during sampling.
    ...
    # Arguments
    - `::T`: Data type of a diffusion
    - `ρs`: ladder of memory parameters for the preconditioned Crank-Nicolson
    - `λs`: ladder of weights that balance between initial choice of auxiliary
            law and the adaptive law based on the mean trajectory
    - `sizes_of_path_coll`: ladder giving the number of paths that are to be
                            used for computing the mean trajectory
    - `skip`: save 1 in every ... many sampled paths
    ...
    """
    function Adaptation(::T, ρs::Vector{S}, λs, sizes_of_path_coll, skip=1) where {T,S}
        TV = Val{true}
        M = maximum(sizes_of_path_coll)
        X = [[zeros(T,0)] for i in 1:M]
        N = [1,1]
        new{TV,T,S}(X, ρs, λs, sizes_of_path_coll, skip, N)
    end

    """
        Adaptation{TV,T}()

    Empty constructor.
    """
    Adaptation{TV,T,S}() where {TV,T,S} = new{TV,T,S}()
end

"""
    NoAdaptation()

Helper function for constructing a flag saying that no adaptation is to be done
"""
NoAdaptation() = Adaptation{Val{false},Nothing,Nothing}()

"""
    check_if_adapt(::Adaptation{Val{T}})

Check if any adaptation needs to be done
"""
check_if_adapt(::Adaptation{Val{T}}) where T = T

"""
    still_adapting(adpt::Adaptation{Val{true}})

If the adaptation has not been completed then do nothing, else return a flag
that no further adaptation is to be done
"""
function still_adapting(adpt::Adaptation{Val{true}})
    adpt.N[1] > length(adpt.sizes) ? NoAdaptation() : adpt
end

"""
    still_adapting(adpt::Adaptation{Val{false}})

There is nothing to be done for a flag indicating no adaptation
"""
still_adapting(adpt::Adaptation{Val{false}}) = adpt

"""
    resize!(adpt::Adaptation{TV,T}, m, ns::Vector{Int64})

Resize internal containers `X` with paths so that each storage unit consists of
`m` subunits and each of these subunits is a length `ns[i]` vector of type `T`
"""
function resize!(adpt::Adaptation{TV,T}, m, ns::Vector{Int64}) where {TV,T}
    K = length(adpt.X)
    for i in 1:K
        adpt.X[i] = [[zero(T) for _ in 1:ns[i]] for i in 1:m]
    end
end

"""
    add_path!(adpt::Adaptation{Val{true},T}, X::Vector{SamplePath{T}}, i)

Save path `X` into the history stored by `adpt` object. Do so only if the index
`i` of the current update step is not supposed to be skipped.
"""
function add_path!(adpt::Adaptation{Val{true},T}, X::Vector{SamplePath{T}}, i) where T
    if i % adpt.skip == 0
        m = length(X)
        for j in 1:m
            adpt.X[adpt.N[2]][j] .= X[j].yy
        end
    end
end


"""
    add_path!(adpt::Adaptation{Val{false}}, ::Any, ::Any)

Nothing to be done when no adaptation is to be performed
"""
add_path!(adpt::Adaptation{Val{false}}, ::Any, ::Any) = false


"""
    mean_trajectory(adpt::Adaptation{Val{true}})

Compute the mean trajectory from the history of accepted paths stored in
`adpt.X`
"""
function mean_trajectory(adpt::Adaptation{Val{true}})
    X = adpt.X
    num_paths = adpt.sizes[adpt.N[1]]
    num_segments = length(X[1])
    for i in 2:num_paths
        for j in 1:num_segments
            num_pts = length(X[i][j])
            for k in 1:num_pts
                X[1][j][k] += X[i][j][k]
            end
        end
    end
    for j in 1:num_segments
        num_pts = length(X[1][j])
        for k in 1:num_pts
            X[1][j][k] /= num_paths
        end
    end
    X[1]
end


"""
    print_adaptation_info(adpt::Adaptation{Val{false}}, ::Any, ::Any, ::Any)

Nothing to print for no adaptation
"""
print_adaptation_info(adpt::Adaptation{Val{false}}, ::Any, ::Any, ::Any) = nothing


"""
    print_adaptation_info(adpt::Adaptation{Val{true}}, acc_imp_counter,
                          acc_updt_counter, i)

Print some information regarding acceptance rate during adaptation to the
console
"""
function print_adaptation_info(adpt::Adaptation{Val{true}}, acc_imp_counter,
                               accUpdtCounter, i)
    if i % adpt.skip == 0 && adpt.N[2] == adpt.sizes[adpt.N[1]]
        print("--------------------------------------------------------\n")
        print(" Adapting...\n")
        print(" Using ", adpt.N[2], " many paths, thinned by ", adpt.skip, "\n")
        print(" Previous imputation acceptance rate: ", acc_imp_counter/i, "\n")
        print(" Previous param update acceptance rate: ", acc_updt_counter./i, "\n")
        print("--------------------------------------------------------\n")
    end
end
