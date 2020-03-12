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
θ₀ = [0.1, 0.05, 0.1, 0.4, 0.01, 0.02]

Pˣ = EcoFinEq(θ₀...)

P̃ = [EcoFinEqAux(θ₀..., t₀, u, T, v) for (t₀,T,u,v)
     in zip(obs_time[1:end - 1], obs_time[2:end], obs[1:end - 1], obs[2:end])]



model_setup = DiffusionSetup(Pˣ,P̃, PartObs())

L1 = @SMatrix [0. 0. 1.]
L2 = @SMatrix [1. 0. 0.;
               0. 1. 0.;
               0. 0. 1.]
1/360
Σdiagel = 0.2/360
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

readj = (100, 0.0001, 0.1, 999.9, 0.234, 50) # readjust_param
readj2 = (100, 0.0001, 0.1, 999.9, 0.234, 50)
#ρ, δ, γ, κ, η, σ
p1 = ParamUpdate(MetropolisHastingsUpdt(), 1, θ₀,
            UniformRandomWalk(1., true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), true) , readj
            )
p2 = ParamUpdate(MetropolisHastingsUpdt(), 4, θ₀,
            UniformRandomWalk(1., true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), true) , readj
            )

mcmc_setup = MCMCSetup(imputation, p1, p2)

schedule = MCMCSchedule(5*10^2, [[1], [2, 3, 2, 3]],
                  (save=10^1, verbose=5*10, warm_up=10,
                   #readjust=(x->x%100==0), fuse=(x->false)))
                   readjust=(x->false), fuse=(x->false)))


Random.seed!(1)
out, chains = mcmc(mcmc_setup, schedule, model_setup)


error("STOP HERE")
using Makie
p1 = scatter(obs_time, [(i-1)%90 == 0 ?  obs[i][3] : obs[i][1] for i in 1:length(obs)],  markersize = 0.1)
for j in (length(out.paths)- 20):length(out.paths)
    lines!(p1, out.time, [out.paths[j][i][3] for i in 1:length(out.paths[1])], color = (:red, 0.01))
end

p2 = scatter(df2.time, df2.x1,  markersize = 0.1)
for j in (length(out.paths)- 20):length(out.paths)
    lines!(p2, out.time, [out.paths[j][i][1] for i in 1:length(out.paths[1])], color = (:red, 0.01))
end

p3 = scatter(df2.time, df2.x2,  markersize = 0.1)
for j in (length(out.paths)- 20):length(out.paths)
    lines!(p3, out.time, [out.paths[j][i][2] for i in 1:length(out.paths[1])], color = (:red, 0.01))
end


p_final = hbox(p3, p2, p1)
resize!(p_final, 5000,3000)
save("output/smoothing.png", p_final)


chain = chains.θ_chain
param_ρ = Makie.lines([chain[i][1] for i in 1:length(chain)])
Makie.lines!(param_ρ, [0.0, length(chain)],[0.03, 0.03], color = (:red, 0.5))
save("output/estimate_rho.png", param_b)


param_κ = Makie.lines([chain[i][4] for i in 1:length(chain)])
Makie.lines!(param_κ, [0.0, length(chain)],[0.2, 0.2], color = (:red, 0.5))
