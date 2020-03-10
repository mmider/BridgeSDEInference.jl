### INFO
# Try to test parameter update.
# Taking a subsample of data
# Increasing the observation noise to Σ= 10^(-2)

using Distributions # to define priors
using Random        # to seed the random number generator
using DataFrames
using CSV
using BridgeSDEInference
using LinearAlgebra
Base.source_dir()
HOME_DIR = joinpath(Base.source_dir(),"..","..")
AUX_DIR = joinpath(HOME_DIR, "src", "auxiliary")
OUT_DIR = joinpath(HOME_DIR, "output")
include(joinpath(AUX_DIR, "read_and_write_data.jl"))
include(joinpath(AUX_DIR, "read_ecofin_equilibrium_data.jl"))
include(joinpath(AUX_DIR, "transforms.jl"))


fptObsFlag = false
filename1 = "ecofin_path_interest_rates.csv"
filename2 = "ecofin_path_indexes.csv"
(df1, df2, x0, obs, obs_time, fpt,
      fptOrPartObs) = readData_ecofin_model(joinpath(OUT_DIR,
                    filename1), joinpath(OUT_DIR, filename2))

# θ = [ρ, δ, γ, κ, η, σ]
θ = [0.03, 0.05, 0.1, 0.2, 0.01, 0.02]

Pˣ = EcoFinEq(θ...)

P̃ = [EcoFinEqAux(θ..., t₀, u, T, v) for (t₀,T,u,v)
     in zip(obs_time[1:end - 1], obs_time[2:end], obs[1:end - 1], obs[2:end])]



model_setup = DiffusionSetup(Pˣ,P̃, PartObs())

L1 = @SMatrix [0. 0. 1.]
L2 = @SMatrix [1. 0. 0.;
               0. 1. 0.;
               0. 0. 1.]

Σdiagel = 10^(-4)
Σ1 = @SMatrix [Σdiagel]
Σ2 = L2*Σdiagel
set_observations!(model_setup, [i%90 == 0 ? L2 : L1 for i in 1:length(obs)-1], [i%90 == 0 ? Σ2 : Σ1 for i in 1:length(obs)-1], obs, obs_time, fpt)


dt = 10^(-4)
set_imputation_grid!(model_setup, dt)

set_x0_prior!(model_setup, KnownStartingPt(x0))


initialise!(eltype(x0), model_setup, Vern7(), false, NoChangePt(100))
set_auxiliary!(model_setup; skip_for_save=10^0, adaptive_prop=NoAdaptation())

pCN = 0.9
imputation = Imputation(NoBlocking(), pCN, Vern7())


mcmc_setup = MCMCSetup(imputation)

schedule = MCMCSchedule(10^4, [[1]],
                  (save=10^2, verbose=5*10^3, warm_up=100,
                   #readjust=(x->x%100==0), fuse=(x->false)))
                   readjust=(x->false), fuse=(x->false)))


Random.seed!(1)
out, chains = mcmc(mcmc_setup, schedule, model_setup)


error("STOP HERE")
using Makie
p1 = scatter(obs_time, [(i-1)%90 == 0 ?  obs[i][3] : obs[i][1] for i in 1:length(obs)],  markersize = 0.05)
for j in (length(out.paths)- 20):length(out.paths)
    lines!(p1, out.time, [out.paths[j][i][3] for i in 1:length(out.paths[1])], color = (:red, 0.1))
end



p2 = scatter(df2.time, df2.x1,  markersize = 0.0)
for j in (length(out.paths)- 20):length(out.paths)
    lines!(p2, out.time, [out.paths[j][i][1] for i in 1:length(out.paths[1])], color = (:red, 0.1))
end

p3 = scatter(df2.time, df2.x2,  markersize = 0.1)
for j in (length(out.paths)- 20):length(out.paths)
    lines!(p3, out.time, [out.paths[j][i][2] for i in 1:length(out.paths[1])], color = (:red, 0.1))
end
