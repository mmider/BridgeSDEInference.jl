using PyPlot
using RollingFunctions


function plot_acceptance(updts::Vector{<:ParamUpdate{MetropolisHastingsUpdt}})
    fig, ax = plt.subplots(2, length(updts), figsize=(3*length(updts), 5))
    for (i,updt) in enumerate(updts)
        ax[1,i].plot(updt.readjust_history, color="steelblue", linewidth=1.0)
        ax[2,i].plot(rollmean(updt.accpt_history, 100), color="steelblue", linewidth=1.0)
        coord = indices(updt.updt_coord)[1]
        print(coord)
        ax[1,i].set_title(string("Parameter ", string(coord)))
    end
    ax[1,1].set_ylabel("dispersion")
    ax[2,1].set_ylabel("acceptance rate")
    plt.tight_layout()
    ax
end

function plot_acceptance(updts::Vector{<:Imputation}, adaptations=nothing, adpt_intv=100)
    max_num_segm = maximum([length(updt.ρs) for updt in updts])
    fig, ax = plt.subplots(2*length(updts), max_num_segm, figsize=(3*max_num_segm, 5*length(updts)))
    if adaptations !== nothing
        adaptations = cumsum(adaptations)
    end
    for (i,updt) in enumerate(updts)
        for j in 1:length(updt.ρs_history)
            ρ = [ρs[j] for ρs in updt.ρs_history]
            ah = [a[j] for a in updt.accpt_history]
            ax[2*(i-1)+1,j].plot(ρ, color="steelblue", linewidth=1.0)
            ax[2*(i-1)+2,j].plot(rollmean(ah, adpt_intv), color="steelblue", linewidth=1.0)
            ylimA = ax[2*(i-1)+1,j].get_ylim()
            ylimB = ax[2*(i-1)+2,j].get_ylim()
            if adaptations !== nothing
                for adpt in adaptations
                    ax[2*(i-1)+1,j].plot([adpt/adpt_intv, adpt/adpt_intv], ylimA, color="red", linestyle="dashed", linewidth=1.0)
                    ax[2*(i-1)+2,j].plot([adpt, adpt], ylimB, color="red", linestyle="dashed", linewidth=1.0)
                end
            end
        end
        ax[2*(i-1)+1,1].set_ylabel("dispersion")
        ax[2*(i-1)+2,1].set_ylabel("acceptance rate")
    end
    plt.tight_layout()
    ax
end




function plot_chains(ws::MCMCWorkspace,indices=nothing;truth=nothing,figsize=(15,10),
                     ylims=nothing)
    θs = ws.θ_chain
    num_steps = length(θs)
    if indices === nothing
        indices = 1:length(θs[1])
    end
    n_chains = length(indices)

    fig, ax = plt.subplots(n_chains, 1, figsize=figsize, sharex=true)

    for (i,ind) in enumerate(indices)
        ax[i].plot([θ[ind] for θ in θs],color="steelblue", linewidth=0.5)
    end
    if truth !== nothing
        for (i,ind) in enumerate(indices)
            ax[i].plot([1,length(θs)],[truth[ind],truth[ind]],color="black",
                       linestyle="--")
        end
    end
    if ylims !== nothing
        for (i,axis) in enumerate(ax)
            if ylims[i] !== nothing
                axis.set_ylim(ylims[i])
            end
        end
    end
    plt.tight_layout()
    ax
end

function plot_paths(ws::Workspace, ws_mcmc::MCMCWorkspace, schedule, coords=nothing; transf=nothing,
                    figsize=(12,8), alpha=0.5, obs=nothing,
                    path_indices=1:length(ws.paths), ylims=nothing, θ_fixed=nothing)
    yy, tt = ws.paths, ws.time
    if coords === nothing
        coords = 1:length(yy[1][1])
    end
    if transf === nothing
        transf = [(x,θ)->x for _ in coords]
    end
    n_coords = length(coords)
    if θ_fixed !== nothing
        θs = fill(θ_fixed, length(yy))
    else
        θs = θs_for_transform(ws_mcmc, schedule)
    end

    fig, ax = plt.subplots(n_coords, 1, figsize=figsize, sharex=true)
    M = min(length(yy), length(path_indices))
    cmap = plt.get_cmap("BuPu")(range(0, 1, length=M)) #"cividis"
    for (i,ind) in enumerate(coords)
        cc = 1
        for j in 1:length(yy)
            if j in path_indices
                path = [transf[i](y, θs[j])[ind] for y in yy[j]]
                ax[i].plot(tt, path, color=cmap[cc,1:end], alpha=alpha, linewidth=0.5)
                cc += 1
            end
        end
    end

    _i = 1
    if obs !== nothing
        for (i,ind) in enumerate(coords)
            if ind in obs.indices
                ax[i].plot(obs.times, [o[_i] for o in obs.vals], mfc="orange",
                           mec="orange", marker="o", linestyle="none",
                           markersize=4)
                _i += 1
            end
        end
    end
    if ylims !== nothing
        for (i,axis) in enumerate(ax)
            if ylims[i] !== nothing
                axis.set_ylim(ylims[i])
            end
        end
    end
    plt.tight_layout()
    ax
end

function θs_for_transform(ws::Any) # old implementation
    θs = ws.θ_chain.θ_chain
    if ws.θ_chain.counter < 2
        θs = [params(ws.P[1].Target) for i in 1:length(ws.paths)]
    else
        warm_up, save_iter = ws.action_tracker.warm_up, ws.action_tracker.save_iter
        num_updts = ws.accpt_tracker.updt_len
        N = Int64((length(θs)-1)/num_updts) + warm_up
        θs = θs[[num_updts*(i-warm_up) for i in 1:N
                 if (i % save_iter == 0) && i > warm_up ]]
    end
    θs
end

function θs_for_transform(ws::MCMCWorkspace, schedule)
    θs = ws.θ_chain
    warm_up, save_iter = schedule.actions.warm_up, schedule.actions.save
    num_updts = length(schedule.updt_idx[1])-1
    N = Int64((length(θs)-1)/num_updts) + warm_up
    θs = θs[[num_updts*(i-warm_up) for i in 1:N
            if (i % save_iter == 0) && i > warm_up ]]
    θs
end
