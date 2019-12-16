### INFO
# Try to test parameter update.
# Taking a subsample of data
# Increasing the observation noise to Σ= 10^(-2)

SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR = joinpath(SRC_DIR, "output")
mkpath(OUT_DIR)
using BridgeSDEInference
include(joinpath(AUX_DIR, "read_and_write_data.jl"))
include(joinpath(AUX_DIR, "read_JR_data.jl"))
include(joinpath(AUX_DIR, "transforms.jl"))

# Using BridgeSDEInference
using Distributions # to define priors
using Random        # to seed the random number generator
using DataFrames
using CSV

# Fetch the data
fptObsFlag = false
filename = "jr_path_part_obs_3n.csv"
init_obs = "jr_initial_obs.csv"
(df, x0, obs, obs_time, fpt,
      fptOrPartObs) = readDataJRmodel(Val(fptObsFlag), joinpath(OUT_DIR, filename))
#take sub set of data


# Initial parameter guess
#θ₀ = [3.25, 100.0, 22.0, 50.0 , 135.0, 5.0, 6.0, 0.56, 0.0, 220.0, 0.0, 0.01, 2000.0, 1.0]
θ₀ = [3.25, 100.0, 22.0, 70.0 , 175.0, 5.0, 6.0, 0.56, 0.0, 150.0, 0.0, 0.01, 1500.0, 1.0]

# P_Target
Pˣ = JRNeuralDiffusion3n(θ₀...)

sigmas = [0.01, 2000.0, 1.0]

#P_auxiliary
P̃ = [JRNeuralDiffusion3nAux(sigmas..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
     in zip(obs_time[1:end - 1], obs_time[2:end], obs[1:end - 1], obs[2:end])]

model_setup = DiffusionSetup(Pˣ,P̃, PartObs())

# Observation operator and noise
L = @SMatrix [0. 1. -1. 0. 0. 0.]
Σdiagel = 10^(-2)
Σ = @SMatrix [Σdiagel]
set_observations!(model_setup, [L for _ in P̃], [Σ for _ in P̃], obs, obs_time, fpt)

# Obsevration frequency
obs_time[2] - obs_time[1]

# Imputation grid < observation frequency
dt = 0.0001
set_imputation_grid!(model_setup, dt)

# Set prior on initial observation
x0 = ℝ{6}(0.08, 18, 15, -0.5, 0, 0)
set_x0_prior!(model_setup, GsnStartingPt(zero(x0), one(SMatrix{6,6,Float64})), 2*x0)


initialise!(eltype(x0), model_setup, Vern7(), false, NoChangePt(100))
set_auxiliary!(model_setup; skip_for_save=10^0, adaptive_prop=NoAdaptation())

pCN = 0.99
imputation = Imputation(NoBlocking(), pCN, Vern7())
readj = (100, 0.005, 0.0001, 999.9, 0.234, 50) # readjust_param
readj2 = (100, 0.00, 0.0001, 999.9, 0.234, 50)
p1 = ParamUpdate(MetropolisHastingsUpdt(), 4, θ₀,
            UniformRandomWalk(0.1, true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), false), readj
            )
p2 = ParamUpdate(MetropolisHastingsUpdt(), 10, θ₀,
            UniformRandomWalk(0.1, true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), false), readj
            )
p3 = ParamUpdate(MetropolisHastingsUpdt(), 5, θ₀,
            UniformRandomWalk(0.1, true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), false), readj
            )
p4 = ParamUpdate(MetropolisHastingsUpdt(), 13, θ₀,
            UniformRandomWalk(0.006, true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), true), readj2
            )


mcmc_setup = MCMCSetup(imputation, p1, p2, p3, p4)
#mcmc_setup = MCMCSetup(imputation, p1, p2, p3, p4)


#mcmc_setup = MCMCSetup(imputation, p4)
#readjust=(x->x%100==0)
schedule = MCMCSchedule(3*10^3, [[1], [2,3,4], [5]],
                  (save=10^2, verbose=10^3, warm_up=100,
                   readjust=(x->x%100==0), fuse=(x->false)))


Random.seed!(4)
out, chains = mcmc(mcmc_setup, schedule, model_setup)
# out.paths
# length(0:0.0001:obs_time[100])
error("STOP HERE")
###PLOTTING###
using Plots
out.paths[end][1]
#ℝ{6}(0.08, 18, 15, -0.5, 0, 0)

[mean(chains.updates[i].accpt_history) for i in 2:5]

display(plot(chain[:][13]))
plot(getindex.(chain, 10))
chains.updates_hist
unique(getindex.(chain, 13))

using Makie

p1 = scatter(obs_time, [obs[i][1] for i in 1:length(obs)],  markersize = 0.05)
lines!(p1, out.time, [out.paths[end][i][2] - out.paths[end][i][3] for i in 1:length(out.paths[1])], color = (:red, 0.1))

p2 = Scene()
lines!(p2, out.time, [out.paths[end][i][1]  for i in 1:length(out.paths[1])], color = (:red, 0.1))

p3 = Scene()
lines!(p3, out.time, [out.paths[end][i][4]  for i in 1:length(out.paths[1])], color = (:red, 0.1))
p4 = Scene()
lines!(p4, out.time, [out.paths[end][i][5]  for i in 1:length(out.paths[1])], color = (:red, 0.1))


p5 = Scene()
lines!(p5, out.time, [out.paths[end][i][6]  for i in 1:length(out.paths[1])], color = (:red, 0.1))


pscene1 = hbox(
    vbox(p3, p5),
    p4,
    vbox(p1, p2)
)
#Makie.save("../../assets/reconstructed_data_with_parameter_estimation1234.png", pscene1)

chain = chains.θ_chain
param_b = Makie.lines([chain[i][4] for i in 1:length(chain)])
param_μy = Makie.lines([chain[i][10] for i in 1:length(chain)])
param_C = Makie.lines([chain[i][5] for i in 1:length(chain)])
param_σy = Makie.lines([chain[i][13] for i in 1:length(chain)])


pscene2 = hbox(
    vbox(param_b, param_μy),
    vbox(param_C, param_σy)
)
#Makie.save("../../assets/parameter_estimation1234.png", pscene2)
