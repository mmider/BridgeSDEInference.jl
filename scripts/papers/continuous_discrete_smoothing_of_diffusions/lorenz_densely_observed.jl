SRC_DIR = joinpath(Base.source_dir(), "..", "..", "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "..", "output")
mkpath(OUT_DIR)

include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
using Main.BridgeSDEInference
#include(joinpath(SRC_DIR, "BridgeSDEInference_for_tests.jl"))


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
function _prepare_setup(ρ=0.96, σ_RW_step=0.5, num_mcmc_steps=4*10^3,
                        do_blocking=false, b_step=2, num_steps_for_change_pt=10,
                        thin_path=10^0, save_path_every=1*10^3)
    setup = MCMCSetup(Pˣ, P̃, PartObs())
    set_observations!(setup, [L for _ in P̃], [Σ for _ in P̃], obs_vals, obs_time) # uses default fpt
    set_imputation_grid!(setup, 1/2000)
    set_transition_kernels!(setup,
                            [RandomWalk([], []),
                             #RandomWalk([2.0, 1.0, 0.64, σ_RW_step], 4)
                             ],
                            ρ, true, [[1,2,3]],#[4]],
                            (ConjugateUpdt(),
                             #MetropolisHastingsUpdt()
                            ),                           # update types
                            Adaptation(x0,
                                       [0.7, 0.4, 0.2, 0.2, 0.2],
                                       [500, 500, 500, 500, 500],
                                       1)
                            )
    set_priors!(setup,
                Priors((MvNormal([0.0,0.0,0.0], diagm(0=>[1000.0, 1000.0, 1000.0])),
                        #ImproperPrior()
                        )),               # priors over parameters
                GsnStartingPt(x0, @SMatrix [400.0 0.0 0.0;
                                            0.0 20.0 0.0;
                                            0.0 0.0 20.0]), # prior over starting point
                x0
                )
    set_mcmc_params!(setup,
                     num_mcmc_steps,    # number of mcmc steps
                     save_path_every,   # save path every ... iteration
                     10^2,              # print progress message every ... iteration
                     thin_path,         # thin the path imputatation points for save
                     100                # number of first iterations without param update
                     )
    if do_blocking
        set_blocking!(setup, ChequeredBlocking(),
                      (collect(1:length(obs_vals)-2)[1:b_step:end], 10^(-10),
                       SimpleChangePt(num_steps_for_change_pt)))
    end
    set_solver!(setup, Vern7(), NoChangePt())
    initialise!(eltype(x0), setup)
    setup
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

skip = 50
obs_time, obs_vals = XX.tt[1:skip:end], [rand(Gaussian(L*x, Σ)) for x in XX.yy[1:skip:end]]

θ_init = [5.0, 15.0, 6.0, 3.0]
aux_flag = Val{(false,true,true)}()
P̃ = [LorenzCVAux(θ_init..., t₀, u, T, v, aux_flag, x0[3]) for (t₀, T, u, v)
     in zip(obs_time[1:end-1], obs_time[2:end], obs_vals[1:end-1], obs_vals[2:end])]
Pˣ = LorenzCV(θ_init...)
print("mid-point time: ", XX.tt[div(length(XX.yy),2)+1], ", mid-point value: ",
      XX.yy[div(length(XX.yy),2)+1], "\n")
print("three-quarters time: ", XX.tt[3*div(length(XX.yy),4)+1], ", three-quarters value: ",
      XX.yy[3*div(length(XX.yy),4)+1], "\n")

#==============================================================================
                    Run the expriment: very large number of blocks
==============================================================================#
setup = _prepare_setup(0.1, 0.3, 1*10^4, true, 2, 15, 1, 10^2)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup)
display(out.accpt_tracker)

plot_chains(out; truth=[10.0, 28.0, 8.0/3.0, 3.0],
            ylims=[nothing, (25,30), (2,5), (0,10)])
plot_paths(out; obs=(times=obs_time[2:end],
                     vals=[[v[1] for v in obs_vals[2:end]],
                           [v[2] for v in obs_vals[2:end]]], indices=[2,3]))

save_paths(out.time, out.paths, "many_obs_many_blocks_paths.csv")
save_param_chain(out.θ_chain.θ_chain, "many_obs_many_blocks_chain.csv")

# repeat, this time save marginals
setup = _prepare_setup(0.1, 0.3, 1*10^4, true, 2, 15, 1000, 1)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup)
display(out.accpt_tracker)
save_marginals(out.time, out.paths, "many_obs_many_blocks_marginals.csv", [1,50,100,150,200])

#==============================================================================
                    Run the expriment: longer blocks (less of them)
==============================================================================#
setup = _prepare_setup(0.5, 0.8, 1*10^4, true, 10, 15, 1, 10^2)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup)
display(out.accpt_tracker)

plot_chains(out; truth=[10.0, 28.0, 8.0/3.0, 3.0],
            ylims=[nothing, (25,30), (2,5), (0,10)])
plot_paths(out; obs=(times=obs_time[2:end],
                     vals=[[v[1] for v in obs_vals[2:end]],
                           [v[2] for v in obs_vals[2:end]]], indices=[2,3]))

save_paths(out.time, out.paths, "many_obs_medium_blocks_paths.csv")
save_param_chain(out.θ_chain.θ_chain, "many_obs_medium_blocks_chain.csv")

setup = _prepare_setup(0.5, 0.8, 1*10^4, true, 4, 15, 1000, 1)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup)
display(out.accpt_tracker)
save_marginals(out.time, out.paths, "many_obs_medium_blocks_marginals.csv", [1,50,100,150,200])

#==============================================================================
                    Run the expriment: very long blocks
==============================================================================#
setup = _prepare_setup(0.2, 0.8, 1*10^4, false, 40, 15, 1, 10^2)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup)
display(out.accpt_tracker)

plot_chains(out; truth=[10.0, 28.0, 8.0/3.0, 3.0],
            ylims=[nothing, (25,30), (2,5), (0,10)])
plot_paths(out; obs=(times=obs_time[2:end],
                     vals=[[v[1] for v in obs_vals[2:end]],
                           [v[2] for v in obs_vals[2:end]]], indices=[2,3]))

save_paths(out.time, out.paths, "many_obs_long_blocks_paths.csv")
save_param_chain(out.θ_chain.θ_chain, "many_obs_long_blocks_chain.csv")

setup = _prepare_setup(0.2, 0.8, 1*10^4, false, 4, 15, 1000, 1)
Random.seed!(4)
out, elapsed = @timeit mcmc(setup)
display(out.accpt_tracker)
save_marginals(out.time, out.paths, "many_obs_long_blocks_marginals.csv", [1,50,100,150,200])
