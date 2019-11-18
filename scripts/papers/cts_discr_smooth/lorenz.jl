SRC_DIR = joinpath(Base.source_dir(), "..", "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "..", "output4")
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
                        do_blocking=false, b_step=2, num_steps_for_change_pt=10,
                        thin_path=10^0, save_path_every=1*10^3)
    model_setup = DiffusionSetup(Pˣ, P̃, PartObs())
    set_observations!(model_setup, [L for _ in P̃], [Σ for _ in P̃], obs_vals,
                      obs_time)
    set_imputation_grid!(model_setup, 1/500)
    set_x0_prior!(model_setup,
                  GsnStartingPt(x0, @SMatrix [400.0 0.0 0.0;
                                                0.0 20.0 0.0;
                                                0.0 0.0 20.0]), # prior over starting point
                  x0)
    set_auxiliary!(model_setup; skip_for_save=thin_path,
                   adaptive_prop=Adaptation(x0, [0.7, 0.4, 0.2, 0.2, 0.2],
                                                [500, 500, 500, 500, 500],
                                            1))
    initialise!(eltype(x0), model_setup, Vern7(), false, NoChangePt(num_steps_for_change_pt))

    blocks = (NoBlocking(), NoBlocking())
    if do_blocking
        blocks = create_blocks( ChequeredBlocking(), model_setup.P,
                                (knots=collect(1:length(obs_vals)-2)[1:b_step:end],
                                ϵ=10^(-10),
                                change_pt=SimpleChangePt(num_steps_for_change_pt)))
    end
    mcmc_setup = MCMCSetup(
        Imputation(blocks[1], ρ, Vern7()),
        Imputation(blocks[2], ρ, Vern7()),
        ParamUpdate(ConjugateUpdt(), [1,2,3], fill(0.0, 4), nothing,
                    MvNormal(fill(0.0, 3), diagm(0=>fill(1000.0, 3))),
                    UpdtAuxiliary(Vern7(), check_if_recompute_ODEs(P̃, [1,2,3])))
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

#==============================================================================
                        Generate densely observed process
==============================================================================#

Random.seed!(4)
θˣ = [10.0, 28.0, 8.0/3.0, 3.0]
Pˣ = LorenzCV(θˣ...)

x0, dt, T = ℝ{3}(1.5, -1.5, 25.0), 1/5000, 2.0
tt = 0.0:dt:T
XX, _ = simulateSegment(ℝ{3}(0.0, 0.0, 0.0), x0, Pˣ, tt)

Σdiagel = 5*10^0
Σ = SMatrix{2,2}(1.0I)*Σdiagel
L = @SMatrix[0.0 1.0 0.0;
             0.0 0.0 1.0]

skip_me = 50
obs_time, obs_vals = XX.tt[1:skip_me:end], [rand(Gaussian(L*x, Σ)) for x in XX.yy[1:skip_me:end]]

θ_init = [5.0, 15.0, 6.0, 3.0]
aux_flag = Val{(false,true,true)}()
P̃ = [LorenzCVAux(θ_init..., t₀, u, T, v, aux_flag, x0[3]) for (t₀, T, u, v)
     in zip(obs_time[1:end-1], obs_time[2:end], obs_vals[1:end-1], obs_vals[2:end])]
Pˣ = LorenzCV(θ_init...)
print("mid-point time: ", XX.tt[div(length(XX.yy),2)+1], ", mid-point value: ",
      XX.yy[div(length(XX.yy),2)+1], "\n")
print("three-quarters time: ", XX.tt[3*div(length(XX.yy),4)+1], ", three-quarters value: ",
      XX.yy[3*div(length(XX.yy),4)+1], "\n")

save_paths(obs_time, [obs_vals], "many_obs_data.csv")

#==============================================================================
                    Run the expriment: very large number of blocks
==============================================================================#
setup = _prepare_setup([[1,3],[2,3]], 0.1, 1*10^5, true, 1, 6, 1, 10^2)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup...)

#plot_acceptance([out[2].updates[1],out[2].updates[2]], [500, 500, 500, 500, 500])
plot_chains(out[2]; truth=[10.0, 28.0, 8.0/3.0, 3.0],
            ylims=[nothing, (25,30), (2,5), (0,10)])
plot_paths(out[1], out[2], setup[2]; obs=(times=obs_time[2:end],
                     vals=[[v[1] for v in obs_vals[2:end]],
                           [v[2] for v in obs_vals[2:end]]], indices=[2,3]))

save_paths(out[1].time, out[1].paths, "many_obs_many_blocks_paths.csv")
save_param_chain(out[2].θ_chain, "many_obs_many_blocks_chain.csv")
save_history(out[2].updates[1].accpt_history, "many_obs_many_blocks_accpt_hist_1.csv")
save_history(out[2].updates[2].accpt_history, "many_obs_many_blocks_accpt_hist_2.csv")
save_history(out[2].updates[1].ρs_history, "many_obs_many_blocks_rho_hist_1.csv")
save_history(out[2].updates[2].ρs_history, "many_obs_many_blocks_rho_hist_2.csv")

# repeat, this time save marginals
setup = _prepare_setup([[1,3],[2,3]], 0.1, 1*10^5, true, 1, 6, 1000, 1)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup...)
display(out.accpt_tracker)
#_temp = 0
#save_marginals(out[1].time, out[1].paths[1+_temp*10^4:(_temp+1)*10^4],
#               join(["many_obs_many_blocks_marginals_", string(_temp), ".csv"]),
#               [1,50,100,150,200])
save_marginals(out[1].time, out[1].paths, "many_obs_many_blocks_marginals.csv"),
               [1,50,100,150,200])
#==============================================================================
                    Run the expriment: longer blocks (less of them)
==============================================================================#
setup = _prepare_setup([[1,3],[2,3]], 0.5, 1*10^5, true, 4, 6, 1, 10^2)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup...)

plot_chains(out[2]; truth=[10.0, 28.0, 8.0/3.0, 3.0],
            ylims=[nothing, (25,30), (2,5), (0,10)])
plot_paths(out[1], out[2], setup[2]; obs=(times=obs_time[2:end],
                     vals=[[v[1] for v in obs_vals[2:end]],
                           [v[2] for v in obs_vals[2:end]]], indices=[2,3]))

save_paths(out[1].time, out[1].paths, "many_obs_medium_blocks_paths.csv")
save_param_chain(out[2].θ_chain, "many_obs_medium_blocks_chain.csv")
save_history(out[2].updates[1].accpt_history, "many_obs_medium_blocks_accpt_hist_1.csv")
save_history(out[2].updates[2].accpt_history, "many_obs_medium_blocks_accpt_hist_2.csv")
save_history(out[2].updates[1].ρs_history, "many_obs_medium_blocks_rho_hist_1.csv")
save_history(out[2].updates[2].ρs_history, "many_obs_medium_blocks_rho_hist_2.csv")


setup = _prepare_setup([[1,3],[2,3]], 0.5, 1*10^5, true, 4, 6, 1000, 1)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup...)
save_marginals(out[1].time, out[1].paths, "many_obs_medium_blocks_marginals.csv", [1,50,100,150,200])
#_temp = 9
#save_marginals(out[1].time, out[1].paths[1+_temp*10^4:(_temp+1)*10^4],
#               join(["many_obs_medium_blocks_marginals_", string(_temp), ".csv"]),
#               [1,50,100,150,200])
#==============================================================================
                    Run the expriment: very long blocks
==============================================================================#
setup = _prepare_setup([[1,3]], 0.5, 1*10^5, false, 40, 6, 1, 10^2)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup...)
display(out.accpt_tracker)

plot_chains(out[2]; truth=[10.0, 28.0, 8.0/3.0, 3.0],
            ylims=[nothing, (25,30), (2,5), (0,10)])
plot_paths(out[1], out[2], setup[2]; obs=(times=obs_time[2:end],
                     vals=[[v[1] for v in obs_vals[2:end]],
                           [v[2] for v in obs_vals[2:end]]], indices=[2,3]))

save_paths(out[1].time, out[1].paths, "many_obs_long_blocks_paths.csv")
save_param_chain(out[2].θ_chain, "many_obs_long_blocks_chain.csv")
save_history(out[2].updates[1].accpt_history, "many_obs_long_blocks_accpt_hist_1.csv")
save_history(out[2].updates[1].ρs_history, "many_obs_long_blocks_rho_hist_1.csv")

setup = _prepare_setup([[1,3]], 0.5, 1*10^5, false, 40, 6, 1000, 1)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup...)
save_marginals(out[1].time, out[1].paths, "many_obs_long_blocks_marginals.csv", [1,50,100,150,200])
#_temp = 9
#save_marginals(out[1].time, out[1].paths[1+_temp*10^4:(_temp+1)*10^4],
#               join(["many_obs_long_blocks_marginals_", string(_temp), ".csv"]),
#               [1,50,100,150,200])








#==============================================================================
                        Generate sparsely observed process
==============================================================================#

Random.seed!(4)
θˣ = [10.0, 28.0, 8.0/3.0, 3.0]
Pˣ = LorenzCV(θˣ...)

x0, dt, T = ℝ{3}(1.5, -1.5, 25.0), 1/5000, 2.0
tt = 0.0:dt:T
XX, _ = simulateSegment(ℝ{3}(0.0, 0.0, 0.0), x0, Pˣ, tt)

Σdiagel = 5*10^(-2)
Σ = SMatrix{2,2}(1.0I)*Σdiagel
L = @SMatrix[0.0 1.0 0.0;
             0.0 0.0 1.0]

skip_me = 1000
obs_time, obs_vals = XX.tt[1:skip_me:end], [rand(Gaussian(L*x, Σ)) for x in XX.yy[1:skip_me:end]]

θ_init = [5.0, 15.0, 6.0, 3.0]
aux_flag = Val{(false,true,true)}()
P̃ = [LorenzCVAux(θ_init..., t₀, u, T, v, aux_flag, x0[3]) for (t₀, T, u, v)
     in zip(obs_time[1:end-1], obs_time[2:end], obs_vals[1:end-1], obs_vals[2:end])]
Pˣ = LorenzCV(θ_init...)
print("mid-point time: ", XX.tt[div(length(XX.yy),2)+1], ", mid-point value: ",
      XX.yy[div(length(XX.yy),2)+1], "\n")
print("three-quarters time: ", XX.tt[3*div(length(XX.yy),4)+1], ", three-quarters value: ",
      XX.yy[3*div(length(XX.yy),4)+1], "\n")
save_paths(obs_time, [obs_vals], "few_obs_data.csv")

#==============================================================================
                    Run the expriment: very long blocks
==============================================================================#
setup = _prepare_setup([[1,3]], 0.96, 1*10^5, false, 40, 15, 1, 10^2)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup...)

plot_chains(out[2]; truth=[10.0, 28.0, 8.0/3.0, 3.0],
            ylims=[nothing, nothing, nothing, (0,10)])
plot_paths(out[1], out[2], setup[2]; obs=(times=obs_time[2:end],
                     vals=[[v[1] for v in obs_vals[2:end]],
                           [v[2] for v in obs_vals[2:end]]], indices=[2,3]))

save_paths(out[1].time, out[1].paths, "few_obs_long_blocks_paths.csv")
save_param_chain(out[2].θ_chain, "few_obs_long_blocks_chain.csv")
save_history(out[2].updates[1].accpt_history, "few_obs_long_blocks_accpt_hist_1.csv")
save_history(out[2].updates[1].ρs_history, "few_obs_long_blocks_rho_hist_1.csv")

setup = _prepare_setup([[1,3]], 0.96, 1*10^5, false, 4, 15, 1000, 1)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup...)
save_marginals(out[1].time, out[1].paths, "few_obs_long_blocks_marginals.csv", [1,3,6,9,10])
_temp = 9
save_marginals(out[1].time, out[1].paths[1+_temp*10^4:(_temp+1)*10^4],
               join(["few_obs_long_blocks_marginals_", string(_temp), ".csv"]),
               [1,3,6,9,10])



#==============================================================================
                    Run the expriment: very large number of blocks
==============================================================================#
setup = _prepare_setup([[1,3],[2,3]], 0.9, 1*10^5, true, 1, 100, 1, 10^2)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup...)

plot_chains(out[2]; truth=[10.0, 28.0, 8.0/3.0, 3.0],
            ylims=[nothing, (25,30), (2,5), (0,10)])
plot_paths(out[1], out[2], setup[2]; obs=(times=obs_time[2:end],
                     vals=[[v[1] for v in obs_vals[2:end]],
                           [v[2] for v in obs_vals[2:end]]], indices=[2,3]))

save_paths(out[1].time, out[1].paths, "few_obs_many_blocks_paths.csv")
save_param_chain(out[2].θ_chain, "few_obs_many_blocks_chain.csv")
save_history(out[2].updates[1].accpt_history, "few_obs_many_blocks_accpt_hist_1.csv")
save_history(out[2].updates[2].accpt_history, "few_obs_many_blocks_accpt_hist_2.csv")
save_history(out[2].updates[1].ρs_history, "few_obs_many_blocks_rho_hist_1.csv")
save_history(out[2].updates[2].ρs_history, "few_obs_many_blocks_rho_hist_2.csv")


# repeat, this time save marginals
setup = _prepare_setup([[1,3],[2,3]], 0.9, 1*10^5, true, 1, 100, 1000, 1)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup...)
save_marginals(out[1].time, out[1].paths, "few_obs_many_blocks_marginals.csv", [1,3,6,9,10])
_temp = 9
save_marginals(out[1].time, out[1].paths[1+_temp*10^4:(_temp+1)*10^4],
               join(["few_obs_many_blocks_marginals_", string(_temp), ".csv"]),
               [1,3,6,9,10])
#==============================================================================
            Run the expriment: very large number of blocks and adaptive
#NOTE because adaptation is currently implemented in a rather inefficient way
# the code responsible for it is by default **COMMENTED OUT**. Consequently,
# to run the code with adaptation one needs to manually remove `#` from the file
# `src/examples/lorenz_system_const_vola.jl` in line 104--105. Run the code
# below and comment it out again.
==============================================================================#
setup = _prepare_setup([[1,3],[2,3]], 0.9, 1*10^5, true, 1, 100, 1, 10^2)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup...)

plot_chains(out[2]; truth=[10.0, 28.0, 8.0/3.0, 3.0],
            ylims=[nothing, (25,30), (2,5), (0,10)])
plot_paths(out[1], out[2], setup[2]; obs=(times=obs_time[2:end],
                     vals=[[v[1] for v in obs_vals[2:end]],
                           [v[2] for v in obs_vals[2:end]]], indices=[2,3]))

save_paths(out[1].time, out[1].paths, "few_obs_many_blocks_and_adpt_paths.csv")
save_param_chain(out[2].θ_chain, "few_obs_many_blocks_and_adpt_chain.csv")
save_history(out[2].updates[1].accpt_history, "few_obs_many_blocks_and_adpt_accpt_hist_1.csv")
save_history(out[2].updates[2].accpt_history, "few_obs_many_blocks_and_adpt_accpt_hist_2.csv")
save_history(out[2].updates[1].ρs_history, "few_obs_many_blocks_and_adpt_rho_hist_1.csv")
save_history(out[2].updates[2].ρs_history, "few_obs_many_blocks_and_adpt_rho_hist_2.csv")



setup = _prepare_setup([[1,3],[2,3]], 0.9, 1*10^5, true, 1, 100, 1000, 1)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup...)
save_marginals(out[1].time, out[1].paths, "few_obs_many_blocks_and_adpt_marginals.csv", [1,3,6,9,10])
_temp = 9
save_marginals(out[1].time, out[1].paths[1+_temp*10^4:(_temp+1)*10^4],
               join(["few_obs_many_blocks_and_adpt_marginals_", string(_temp), ".csv"]),
               [1,3,6,9,10])







#==============================================================================
            Run the expriment: very large number of blocks and adaptive
#NOTE needs to be done by manually uncommenting the definitions of B₀ and β₀
==============================================================================#
setup = _prepare_setup([[1,3]], 0.96, 1*10^5, false, 40, 15, 1, 10^2)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup...)

plot_chains(out[2]; truth=[10.0, 28.0, 8.0/3.0, 3.0],
            ylims=[nothing, nothing, nothing, (0,10)])
plot_paths(out[1], out[2], setup[2]; obs=(times=obs_time[2:end],
                     vals=[[v[1] for v in obs_vals[2:end]],
                           [v[2] for v in obs_vals[2:end]]], indices=[2,3]))

save_paths(out[1].time, out[1].paths, "few_obs_long_blocks_zero_drift_paths.csv")
save_param_chain(out[2].θ_chain, "few_obs_long_blocks_zero_drift_chain.csv")
save_history(out[2].updates[1].accpt_history, "few_obs_long_blocks_zero_drift_accpt_hist_1.csv")
save_history(out[2].updates[1].ρs_history, "few_obs_long_blocks_zero_drift_rho_hist_1.csv")

setup = _prepare_setup([[1,3]], 0.96, 1*10^5, false, 4, 15, 1000, 1)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup...)
save_marginals(out[1].time, out[1].paths, "few_obs_long_blocks_zero_drift_marginals.csv", [1,3,6,9,10])
_temp = 9
save_marginals(out[1].time, out[1].paths[1+_temp*10^4:(_temp+1)*10^4],
               join(["few_obs_long_blocks_zero_drift_marginals_", string(_temp), ".csv"]),
               [1,3,6,9,10])
