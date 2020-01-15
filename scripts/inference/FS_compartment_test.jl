using BridgeSDEInference
using Distributions # to define priors
using Random        # to seed the random number generator
using DataFrames
using CSV
using RCall


SRC_DIR = joinpath(Base.source_dir(), "..", "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
mkpath(OUT_DIR)
include(joinpath(AUX_DIR, "read_and_write_data.jl"))
include(joinpath(AUX_DIR, "read_JR_data.jl"))
include(joinpath(AUX_DIR, "transforms.jl"))


fptObsFlag = false
filename = "FS_testdata.csv"
(df, x0, obs, obs_time, fpt,
      fptOrPartObs) = readDataJRmodel(Val(fptObsFlag), joinpath(OUT_DIR, filename))
#take sub set of data

θ₀ = [117, 5.83, 1.25, 1.5, 1.41, 0.0] # about true
θ₀ = [117, 1.83, 3.25, 10.5, 3.41, 0.0] # only beta wrong
#θ₀ = [100.0, 5.83, 1.25, 4*1.5, 4*1.41, 0.0] #+ fill(4.0,6)
#θ₀ = [100., 1.0, 1.0, 1.0, 1.0, 1.0]

# P_Target
Pˣ = FS(θ₀...)

#P_auxiliary
P̃ = [FSAux(θ₀..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
     in zip(obs_time[1:end - 1], obs_time[2:end], obs[1:end - 1], obs[2:end])]

model_setup = DiffusionSetup(Pˣ,P̃, PartObs())
# Observation operator and noise
L = @SMatrix [.5 .5 ]
Σdiagel = 10^(-2)
Σ = @SMatrix [Σdiagel]
set_observations!(model_setup, [L for _ in P̃], [Σ for _ in P̃], obs, obs_time, fpt)


# Imputation grid < observation frequency
dt = 0.01
set_imputation_grid!(model_setup, dt)

# Set prior on initial observation
x0 = ℝ{2}(0.0, 0.0)
#set_x0_prior!(model_setup, GsnStartingPt(zero(x0), one(SMatrix{2,2,Float64})), 2*x0)
set_x0_prior!(model_setup, KnownStartingPt(x0))
initialise!(eltype(x0), model_setup, Vern7(), false, NoChangePt(100))
set_auxiliary!(model_setup; skip_for_save=10^0, adaptive_prop=NoAdaptation())

pCN = 0.02
imputation = Imputation(NoBlocking(), pCN, Vern7())

readj = (100, 0.001, 0.001, 999.9, 0.234, 50) # readjust_param
readj2 = (100, 0.001, 0.001, 999.9, 0.234, 50)
readj3 = (100, 0.001, 0.001, 999.9, 0.234, 50)

p1 = ParamUpdate(MetropolisHastingsUpdt(), 1, θ₀,
            UniformRandomWalk(.05, true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), true), readj
            )
p2 = ParamUpdate(MetropolisHastingsUpdt(), 2, θ₀,
            UniformRandomWalk(.05, true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), true), readj
            )
p3 = ParamUpdate(MetropolisHastingsUpdt(), 3, θ₀,
            UniformRandomWalk(.05, true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), true), readj3
            )
p4 = ParamUpdate(MetropolisHastingsUpdt(), 4, θ₀,
            UniformRandomWalk(.02, true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), true), readj2
            )
p5 = ParamUpdate(MetropolisHastingsUpdt(), 5, θ₀,
            UniformRandomWalk(.1, true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), true), readj2
            )



#mcmc_setup = MCMCSetup(imputation, p1, p2, p3, p4, p5)
# schedule = MCMCSchedule(10^4, [[1], [2,3,4,5,6]],
#                   (save=10^2, verbose=10^3, warm_up=100,
#                    readjust=(x->x%100==0), fuse=(x->false)))


mcmc_setup = MCMCSetup(imputation, p2,p3, p4, p5)
schedule = MCMCSchedule(10^4, [[1],[2],[3],[4],[5]],
                  (save=10^2, verbose=10^3, warm_up=100,
                   readjust=(x->x%100==0), fuse=(x->false)))



Random.seed!(4)
out, chains = mcmc(mcmc_setup, schedule, model_setup)

ec(x,i) = map(y -> y[i], x)
chain = chains.θ_chain
df = DataFrame(iterate = 1:length(chain), alpha = ec(chain,1),
beta = ec(chain,2), lambda = ec(chain,3), mu = ec(chain,4), sigma1 = ec(chain,5))

@rput df
R"""
library(ggplot2)
library(tidyverse)
df %>% gather(key='component',value='value', alpha, beta, lambda, mu, sigma1) %>%
 ggplot() + geom_path(aes(x=iterate,y=value)) +
 facet_wrap(~component,scales='free') + theme_light()
"""


println("true vals:  117, 5.83, 1.25, 1.5, 1.41")

error("STOP HERE")



using Makie

p1 = Makie.scatter(obs_time, [obs[i][1] for i in 1:length(obs)],  markersize = 0.05)
lines!(p1, out.time, [out.paths[end][i][1]*.5 + out.paths[end][i][2]*0.5 for i in 1:length(out.paths[1])], color = (:red, 0.1))

p2 = Scene()
lines!(p2, out.time, [out.paths[end][i][1]  for i in 1:length(out.paths[1])], color = (:red, 0.1))

p3 = Scene()
lines!(p3, out.time, [out.paths[end][i][2]  for i in 1:length(out.paths[1])], color = (:red, 0.1))


pscene1 = hbox(
        p3,
    vbox(p1, p2)
)


param_α = Makie.lines([chain[i][1] for i in 1:length(chain)])
param_β = Makie.lines([chain[i][2] for i in 1:length(chain)])
param_λ = Makie.lines([chain[i][3] for i in 1:length(chain)])
param_μ = Makie.lines([chain[i][4] for i in 1:length(chain)])
param_σ1 = Makie.lines([chain[i][5] for i in 1:length(chain)])



pscene2 = hbox(
    vbox(param_α  , param_β),
    vbox(param_λ, param_σ1)
)


#  117, 5.83, 1.25, 1.5, 1.41
