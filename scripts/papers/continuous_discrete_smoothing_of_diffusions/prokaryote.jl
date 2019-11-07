SRC_DIR = joinpath(Base.source_dir(), "..", "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "..", "output")
mkpath(OUT_DIR)

#include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
#using Main.BridgeSDEInference
include(joinpath(SRC_DIR, "BridgeSDEInference_for_tests.jl"))


using StaticArrays, LinearAlgebra, GaussianDistributions
using Distributions
using Random
using Bridge
using CSV, DataFrames
DIR = "auxiliary"
include(joinpath(SRC_DIR, DIR, "data_simulation_fns.jl"))
include(joinpath(SRC_DIR, DIR, "utility_functions.jl"))
include(joinpath(SRC_DIR, DIR, "plotting_fns.jl"))

#=============================================================================
                Routines for setting up the MCMC sampler
=============================================================================#
function _prepare_setup(updt_order, ρ=0.96, num_mcmc_steps=4*10^3,
                        thin_path=10^0, save_path_every=1*10^3)
    model_setup = DiffusionSetup(P˟, P̃, PartObs())
    set_observations!(model_setup, [L for _ in P̃], [Σ for _ in P̃], obs_vals,
                      obs_time)
    set_imputation_grid!(model_setup, 1/100)
    set_x0_prior!(model_setup,
                  GsnStartingPt(x0, SArray{Tuple{4,4}, Float64}(I)), # prior over starting point
                  x0)
    set_auxiliary!(model_setup; skip_for_save=thin_path,
                   adaptive_prop=Adaptation(x0, fill(0.0, 10), fill(100, 10), 1))
    initialise!(eltype(x0), model_setup, Vern7(), false, NoChangePt())

    mcmc_setup = MCMCSetup(
        Imputation(NoBlocking(), ρ, Vern7()),
        ParamUpdate(MetropolisHastingsUpdt(), [1], fill(0.0, 8),
                    UniformRandomWalk(0.5, true), ImproperPosPrior(),
                    UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, [1]))),
        ParamUpdate(MetropolisHastingsUpdt(), [2], fill(0.0, 8),
                    UniformRandomWalk(0.5, true), ImproperPosPrior(),
                    UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, [2]))),
        ParamUpdate(MetropolisHastingsUpdt(), [4], fill(0.0, 8),
                    UniformRandomWalk(0.5, true), ImproperPosPrior(),
                    UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, [3]))),
        ParamUpdate(MetropolisHastingsUpdt(), [8], fill(0.0, 8),
                    UniformRandomWalk(0.5, true), ImproperPosPrior(),
                    UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, [7]))),

    )
    schedule = MCMCSchedule(num_mcmc_steps, updt_order,
                            (save=save_path_every, verbose=10^2, warm_up=0,
                             readjust=(x->x%100==0), fuse=(x->false)) )
    mcmc_setup, schedule, model_setup
end


#==============================================================================
                        Auxiliary routines for saving
==============================================================================#

function save_paths(tt, paths, filename)
    d = length(paths[1][1])
    xx = copy(tt)
    for i in 1:length(paths)
        for j in 1:d
            xx = hcat(xx, [p[j] for p in paths[i]])
        end
    end
    CSV.write(joinpath(OUT_DIR, filename), DataFrame(xx))
end


function save_marginals(tt, paths, filename, indices)
    d = length(paths[1][1])
    xx = copy(tt[indices])
    N = length(paths)
    stride = div(N, 100)
    for i in 1:length(paths)
        if i % stride == 0
            print(div(i, 100), "% done...\n")
        end
        for j in 1:d
            xx = hcat(xx, [p[j] for p in paths[i][indices]])
        end
    end
    CSV.write(joinpath(OUT_DIR, filename), DataFrame(xx'))
end

function save_param_chain(chain, filename)
    d = length(chain[1])
    xx = [c[1] for c in chain]
    for i in 2:d
        xx = hcat(xx, [c[i] for c in chain])
    end
    CSV.write(joinpath(OUT_DIR, filename), DataFrame(xx))
end


function save_obs(tt, paths, filename)
    d = length(paths[1][1])
    xx = copy(tt)
    for i in 1:length(paths)
        for j in 1:d
            xx = hcat(xx, [p[j] for p in paths[i]])
        end
    end
    CSV.write(joinpath(OUT_DIR, filename), DataFrame(xx))
end

function save_history(to_save, filename)
    n, k = length(to_save), length(to_save[1])
    xx = zeros(n, k)
    for i in 1:n
        xx[i,1:end] .= to_save[i]
    end
    CSV.write(joinpath(OUT_DIR, filename), DataFrame(xx))
end


#================================================================================
                            Generate the process
==============================================================================#

_obs = open(joinpath(OUT_DIR, "prokaryote_custom.dat")) do f
    [map(x->parse(Float64, x), split(l, ' '))[[2,3]] for (i,l) in enumerate(eachline(f)) if i != 1]
end
_obs_time = collect(range(1.0, length(_obs), step=1))

obs_vals = vcat([0.0],[o[2] for o in _obs])
obs_time = vcat(0.0, _obs_time)

# let's start from the true values that generated the data
θ_init = [0.1, 0.7, 0.35, 0.2, 0.1, 0.9, 0.3, 0.1]
K = 10.0
P˟ = Prokaryote(θ_init..., K)
x0 = ℝ{4}([8.0,8.0,8.0,5.0])

auxFlag = Val{:custom}()
start_v = @SVector[5.0, 5.0, 5.0, 5.0]
P̃ = [ProkaryoteAux(θ_init..., K, t₀, u, T, v, auxFlag, start_v) for (t₀, T, u, v)
     in zip(obs_time[1:end-1], obs_time[2:end], obs_vals[1:end-1], obs_vals[2:end])]

Σ = @SMatrix[2.0]
L = @SMatrix[0.0 1.0 2.0 0.0]

#=
model_setup = DiffusionSetup(P˟, P̃, PartObs())
set_observations!(model_setup, [L for _ in P̃], [Σ for _ in P̃], obs_vals,
                  obs_time)
set_imputation_grid!(model_setup, 1/100)
set_x0_prior!(model_setup,
              GsnStartingPt(x0, SArray{Tuple{4,4}, Float64}(I)), # prior over starting point
              x0)
set_auxiliary!(model_setup; skip_for_save=1,
               adaptive_prop=Adaptation(x0, fill(0.0, 10), fill(100, 10), 1))
initialise!(eltype(x0), model_setup, Vern7(), false, NoChangePt())
model_setup


mcmc_setup = MCMCSetup(
    Imputation(NoBlocking(), 0.95, Vern7()),
    ParamUpdate(MetropolisHastingsUpdt(), [1], fill(0.0, 8),
                UniformRandomWalk(0.5, true), ImproperPosPrior(),
                UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, [1]))),
    ParamUpdate(MetropolisHastingsUpdt(), [2], fill(0.0, 8),
                UniformRandomWalk(0.5, true), ImproperPosPrior(),
                UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, [2]))),
    ParamUpdate(MetropolisHastingsUpdt(), [3], fill(0.0, 8),
                UniformRandomWalk(0.5, true), ImproperPosPrior(),
                UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, [3]))),
    ParamUpdate(MetropolisHastingsUpdt(), [7], fill(0.0, 8),
                UniformRandomWalk(0.5, true), ImproperPosPrior(),
                UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, [7]))))
schedule = MCMCSchedule(1*10^4, [[1,2,3,4,5]],
                        (save=10^2, verbose=10^2, warm_up=0,
                         readjust=(x->x%100==0), fuse=(x->false)) )
out = mcmc(mcmc_setup, schedule, model_setup)
=#
setup = _prepare_setup([[1,2,3,4,5]], 0.9, 1*10^4, 1, 10^2)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup...)
plot_chains(out[2]; truth=[0.1, 0.7, 0.35, 0.2, 0.1, 0.9, 0.3, 0.1])
plot_paths(out[1], out[2], setup[2])

save_paths(out[1].time, out[1].paths, "prokaryote_paths.csv")
save_param_chain(out[2].θ_chain, "prokaryote_chain.csv")
save_history(out[2].updates[1].accpt_history, "prokaryote_accpt_hist_1.csv")
save_history(out[2].updates[2].accpt_history, "prokaryote_accpt_hist_2.csv")
save_history(out[2].updates[3].accpt_history, "prokaryote_accpt_hist_3.csv")
save_history(out[2].updates[4].accpt_history, "prokaryote_accpt_hist_4.csv")
save_history(out[2].updates[1].ρs_history, "prokaryote_rho_hist_1.csv")
