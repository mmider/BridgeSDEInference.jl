
function order_updates!(updates)
    function _updt_idx(updt)
        idx = indices(updt.updt_coord)
        @assert length(idx) == 1
        idx[1]
    end
    sort!(updates, by=_updt_idx)
end

function fuse_coord(updates)
    indx = [indices(u.updt_coord)[1] for u in updates]
    @assert length(updates) == length(unique(indx))
    indx
end

function fuse_kernels(updates, cov_mat)
    d = size(cov_mat)[1]
    GaussianRandomWalkMix(2.38^2/d*cov_mat,
                          0.1^2/d*Matrix{Float64}(I, size(cov_mat)...),
                          0.05,
                          [u.t_kernel.pos for u in updates])
end

function fuse_priors(updates)
    prior_list = []
    coord_idx_list = []
    for updt in updates
        push!(prior_list, updt.priors.priors...)
        push!(coord_idx_list, updt.priors.coord_idx...)
    end
    Priors(Tuple(prior_list), Tuple(coord_idx_list))
end

function fuse_aux(updates, aux::Vector{<:UpdtAuxiliary}=[u.aux for u in updates])
    recompute_ODEs = any([a.recompute_ODEs for a in aux])
    UpdtAuxiliary(aux[1].solver, recompute_ODEs)
end
