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
struct Adaptation{TV,T,ST}
    X::Vector{Vector{Vector{T}}} # history of paths
    λs::Vector{Float64}          # ladder of weights that balance initial
                                 # auxiliary law and the adaptive law based on
                                 # the mean trajectory
    sizes::Vector{Int64}         # ladder indicating number of paths that are to
                                 # be used for computing mean trajectory
    skip::Int64                  # save 1 in every ... many sampled paths
    N::Vector{Int64}             # counter #1-current position on the ladder
                                 # #2-current index of the last saved path
    solver::ST

    """
        Adaptation(::T, λs, sizes_of_path_coll, skip=1)

    Initialise adaptation. `λs` and `sizes_of_path_coll` are ladders that
    are traversed during sampling.
    ...
    # Arguments
    - `::T`: Data type of a diffusion
    - `λs`: ladder of weights that balance between initial choice of auxiliary
            law and the adaptive law based on the mean trajectory
    - `sizes_of_path_coll`: ladder giving the number of paths that are to be
                            used for computing the mean trajectory
    - `skip`: save 1 in every ... many sampled paths
    ...
    """
    function Adaptation(::T, λs, sizes_of_path_coll, skip=1, solver::ST=Vern7()
                        ) where {T,ST}
        TV = Val{true}
        M = maximum(sizes_of_path_coll)
        X = [[zeros(T,0)] for i in 1:M]
        N = [1,1]
        new{TV,T,ST}(X, λs, sizes_of_path_coll, skip, N, solver)
    end

    """
        Adaptation{TV,T}()

    Empty constructor.
    """
    Adaptation{TV,T,ST}() where {TV,T,ST} = new{TV,T,ST}()
end

"""
    NoAdaptation()

Helper function for constructing a flag saying that no adaptation is to be done
"""
NoAdaptation() = Adaptation{Val{false},Nothing,Nothing}()


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


function adaptation!(ws, adpt::Adaptation, mcmc_iter, ll)
    if still_adapting(adpt) && mcmc_iter % adpt.skip == 0
        add_path!(ws, adpt)
        ll = update!(ws, adpt, ll)
    end
    ll
end

still_adapting(::Adaptation{Val{false}}) = false
still_adapting(adpt::Adaptation{Val{true}}) = adpt.N[1] <= length(adpt.sizes)


function add_path!(ws, adpt::Adaptation)
    m = length(ws.XX)
    for j in 1:m
        adpt.X[adpt.N[2]][j] .= ws.XX[j].yy
    end
end

function update!(ws, adpt::Adaptation, ll)
    if adpt.N[2] == adpt.sizes[adpt.N[1]]
        X_bar = mean_trajectory(adpt)
        m = length(ws.P)
        for j in 1:m
            Pt = recentre(ws.P[j].Pt, ws.XX[j].tt, X_bar[j])
            update_λ!(Pt, adpt.λs[adpt.N[1]])
            ws.P[j] = GuidPropBridge(ws.P[j], Pt)

            Ptᵒ = recentre(ws.Pᵒ[j].Pt, ws.XX[j].tt, X_bar[j])
            update_λ!(Ptᵒ, adpt.λs[adpt.N[1]])
            ws.Pᵒ[j] = GuidPropBridge(ws.Pᵒ[j], Ptᵒ)
        end

        solve_back_rec!(NoBlocking(), adpt.solver, ws.P)
        #solveBackRec!(NoBlocking(), ws.Pᵒ, ST())
        y = ws.XX[1].yy[1]
        z = inv_start_pt(y, ws.x0_prior, ws.P[1])
        set!(ws.z, z)

        for j in 1:m
            inv_solve!(Euler(), ws.XX[j], ws.WW[j], ws.P[j])
        end
        ll = logpdf(ws.x0_prior, y)
        ll += path_log_likhd(obs_scheme(ws), ws.XX, ws.P, 1:m, ws.fpt)
        ll += lobslikelihood(ws.P[1], y)
        adpt.N[2] = 1
        adpt.N[1] += 1
    else
        adpt.N[2] += 1
    end
    ll
end


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
