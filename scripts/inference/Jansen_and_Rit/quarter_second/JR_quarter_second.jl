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

HOME_DIR = joinpath(Base.source_dir(), "..", "..", "..", "..")
AUX_DIR = joinpath(HOME_DIR, "src", "auxiliary")
OUT_DIR = joinpath(HOME_DIR, "output")
include(joinpath(AUX_DIR, "read_and_write_data.jl"))
include(joinpath(AUX_DIR, "read_JR_data.jl"))
include(joinpath(AUX_DIR, "transforms.jl"))

# Using BridgeSDEInference


# Fetch the data
fptObsFlag = false
filename = "jr_path_part_obs_3n.csv"
(df, x0, obs, obs_time, fpt,
      fptOrPartObs) = readDataJRmodel(Val(fptObsFlag), joinpath(OUT_DIR, filename))
#take sub set of data: 1 second
obs = obs[1:64]
obs_time = obs_time[1:64]

# Initial parameter guess
θ_true = [3.25, 100.0, 22.0, 50.0 , 135.0, 5.0, 6.0, 0.56, 0.0, 220.0, 0.0, 0.01, 2000.0, 1.0]
θ₀ =     [3.25, 100.0, 22.0, 50.0 , 135.0, 5.0, 6.0, 0.56, 0.0, 220.0, 0.0, 0.01, 2000.0, 1.0]
# Boolean for parameters to update
#param_bool = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
#θ₀ = θ_true.*(1 .+ 0.5.*param_bool.*randn(length(θ_true)))


# P_Target
Pˣ = JRNeuralDiffusion3n(θ₀...)



#P_auxiliary
sigmas = θ₀[12:14]
P̃ = [JRNeuralDiffusion3nAux(sigmas..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
     in zip(obs_time[1:end - 1], obs_time[2:end], obs[1:end - 1], obs[2:end])]

model_setup = DiffusionSetup(Pˣ,P̃, PartObs())

# Observation operator and noise
L = @SMatrix [0. 1. -1. 0. 0. 0.]
Σdiagel = 10^(-2)
Σ = @SMatrix [Σdiagel]
set_observations!(model_setup, [L for _ in P̃], [Σ for _ in P̃], obs, obs_time, fpt)


# Imputation grid < observation frequency
dt = 1/256*10^(-2)
set_imputation_grid!(model_setup, dt)

one(SMatrix{6,6,Float64}).*0.001
# Set prior on initial observation
x0 = ℝ{6}(0.08, 18, 15, -0.5, 0, 0)
set_x0_prior!(model_setup, GsnStartingPt(x0.*(1 .+ 0.001.*randn(6)), 0.001.*one(SMatrix{6,6,Float64})), x0.*(1 .+ 0.001.*randn(6)))
#set_x0_prior!(model_setup, KnownStartingPt(x0))

initialise!(eltype(x0), model_setup, Vern7(), false, NoChangePt(100))
set_auxiliary!(model_setup; skip_for_save=10^0, adaptive_prop=NoAdaptation())

pCN = 0.9
imputation = Imputation(NoBlocking(), pCN, Vern7())

#readjusted parameter turned off (see MCMCSchedule setup)
readj = (100, 0.0001, 0.01, 999.9, 0.234, 50) # readjust_param
readj2 = (100, 0.0001, 0.01, 999.9, 0.234, 50)

p1 = ParamUpdate(MetropolisHastingsUpdt(), 4, θ₀,
            UniformRandomWalk(0.1, true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), false) , readj
            )

p2 = ParamUpdate(ConjugateUpdt(), 10, θ₀,
            nothing, MvNormal([250.],diagm(0=>[1000.0])),
            UpdtAuxiliary(Vern7(), false) , readj
            )

# p2 = ParamUpdate(MetropolisHastingsUpdt(), 10, θ₀,
#             UniformRandomWalk(0.2, true), ImproperPosPrior(),
#             UpdtAuxiliary(Vern7(), false) , readj
#             )

p3 = ParamUpdate(MetropolisHastingsUpdt(), 5, θ₀,
            UniformRandomWalk(0.05, true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), false), readj2
            )
p4 = ParamUpdate(MetropolisHastingsUpdt(), 13, θ₀,
            UniformRandomWalk(0.1, true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), true), readj2
            )
p5 = ParamUpdate(MetropolisHastingsUpdt(), 2, θ₀,
            UniformRandomWalk(0.05, true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), false) , readj
            )

p6 = ParamUpdate(MetropolisHastingsUpdt(), 1, θ₀,
            UniformRandomWalk(0.05, true), ImproperPosPrior(),
            UpdtAuxiliary(Vern7(), false) , readj
            )
#mcmc_setup = MCMCSetup(imputation, p1, p5, p3, p4)
mcmc_setup = MCMCSetup(imputation, p1, p2, p3, p4, p5)


#mcmc_setup = MCMCSetup(imputation, p4)
#readjust=(x->x%100==0)
schedule = MCMCSchedule(10^4, [[1], [2,3,4,6], [5]],
                  (save=10^2, verbose=5*10^3, warm_up=100,
                   #readjust=(x->x%100==0), fuse=(x->false)))
                   readjust=(x->false), fuse=(x->false)))


Random.seed!(4)
out, chains = mcmc(mcmc_setup, schedule, model_setup)

println("")
#println("acceptance rates for the for parameters are")
#println("A: ", mean(chains.updates[6].accpt_history), " b: ", mean(chains.updates[2].accpt_history), ", μy: ",mean(chains.updates[3].accpt_history),
#        " C: ", mean(chains.updates[4].accpt_history), " σy: ", mean(chains.updates[5].accpt_history))


# out.paths
# length(0:0.0001:obs_time[100])
error("STOP HERE")
###PLOTTING###

using Makie


p1 = scatter(obs_time, [obs[i][1] for i in 1:length(obs)],  markersize = 0.05)

out.paths
for j in (length(out.paths)- 20):length(out.paths)
    lines!(p1, out.time, [out.paths[j][i][2] - out.paths[j][i][3] for i in 1:length(out.paths[1])], color = (:red, 0.1))
end
p2 = Scene()
for j in (length(out.paths)- 20):length(out.paths)
    lines!(p2, out.time, [out.paths[j][i][1]  for i in 1:length(out.paths[1])], color = (:red, 0.5))
end
p3 = Scene()
for j in (length(out.paths)- 20):length(out.paths)
    lines!(p3, out.time, [out.paths[j][i][4]  for i in 1:length(out.paths[1])], color = (:red, 0.5))
end

p4 = Scene()
for j in (length(out.paths)- 20):length(out.paths)
    lines!(p4, out.time, [out.paths[j][i][5]  for i in 1:length(out.paths[1])], color = (:red, 0.5))
end

p5 = Scene()
for j in 1:length(out.paths)
    lines!(p5, out.time, [out.paths[j][i][6]  for i in 1:length(out.paths[1])], color = (:red, 0.5))
end

pscene1 = hbox(
    vbox(p3, p5),
    p4,
    vbox(p1, p2)
)

Makie.save(joinpath(Base.source_dir(), "output", "reconstructed_data_with_parameter_estimation1234.png"), pscene1)



using Makie
chain = chains.θ_chain
param_b = Makie.lines([chain[i][4] for i in 1:length(chain)])
Makie.lines!(param_b, [0.0, length(chain)],[50, 50], color = (:red, 0.5))
param_b = Makie.title(param_b, "param b")
param_μy = Makie.lines([chain[i][10] for i in 1:length(chain)])
Makie.lines!(param_μy, [0.0, length(chain)],[220, 220], color = (:red, 0.5))
param_μy = Makie.title(param_μy, "param μy")
param_C = Makie.lines([chain[i][5] for i in 1:length(chain)])
Makie.lines!(param_C, [0.0, length(chain)],[135, 135], color = (:red, 0.5))
param_C =  Makie.title(param_C, "param C")
param_σy = Makie.lines([chain[i][13] for i in 1:length(chain)])
Makie.lines!(param_σy, [0.0, length(chain)],[2000, 2000], color = (:red, 0.5))
param_σy =  Makie.title(param_σy, "param σy")
param_A = Makie.lines([chain[i][1] for i in 1:length(chain)])
Makie.lines!(param_A, [0.0, length(chain)],[3.25, 3.25], color = (:red, 0.5))
param_A =  Makie.title(param_A, "param A")

param_a = Makie.lines([chain[i][2] for i in 1:length(chain)])
Makie.lines!(param_a, [0.0, length(chain)],[100.0, 100.0], color = (:red, 0.5))
param_a =  Makie.title(param_a, "param a")


pscene2 = hbox(
    vbox(param_μy, param_σy),
    vbox(param_b, param_C),
    vbox(param_A, param_a)
    )

Makie.save(joinpath(Base.source_dir(), "output", "identifibility_fix_mu_y.png"), pscene2)



############### CORRELATIONS #####################################

using AbstractPlotting
using Makie
using ColorSchemes

ratio_a_C =  Makie.lines([chain[i][2]./chain[i][5] for i in 1:length(chain)])
ratio_a_C =  Makie.title(ratio_a_C, "a/C")
ratio_a_b =  Makie.lines([chain[i][2]./chain[i][4] for i in 1:length(chain)])
ratio_a_b =  Makie.title(ratio_a_b, "a/b")

pscene3 = vbox(
    hbox(param_C, param_b,  param_a),
    hbox(ratio_a_C, ratio_a_b)
    )


t = 10000:length(chain) # time steps
b = [chain[i][4] for i in 10000:length(chain)]
C = [chain[i][5] for i in 10000:length(chain)]
a = [chain[i][2] for i in 10000:length(chain)]


p1 = scatter(
    b,
    C,
    color = t,
    colormap = ColorSchemes.magma.colors
    )


p2 = scatter(
    b,
    a,
    color = t,
    colormap = ColorSchemes.magma.colors
    )


p3 = scatter(
    a,
    C,
    color = t,
    colormap = ColorSchemes.magma.colors
    )



cm = colorlegend(
    p1[end],             # access the plot of Scene p1
    raw = true,          # without axes or grid
    camera = campixel!,  # gives a concrete bounding box in pixels
                  # so that the `vbox` gives you the right size
    width = (            # make the colorlegend longer so it looks nicer
        30,              # the width
        540              # the height
        )
    )

axis1 = p1[Axis]
axis1[:names, :axisnames] = ("b", "C")

axis2 = p2[Axis]
axis2[:names, :axisnames] = ("b", "a")

axis3 = p3[Axis]
axis3[:names, :axisnames] = ("a", "C")

scena_finale = vbox(p1,p2,p3, cm)

Makie.save(joinpath(Base.source_dir(), "output", "parameter_correlations1.png"), scena_finale)





##### 3d plots


t = 10000:length(chain) # time steps
b = [chain[i][4] for i in 10000:length(chain)]
C = [chain[i][5] for i in 10000:length(chain)]
a = [chain[i][2] for i in 10000:length(chain)]

p1 = scatter(
    x,
    y,
    z,
    color = t,
    colormap = ColorSchemes.magma.colors
    )


cm = colorlegend(
    p1[end],             # access the plot of Scene p1
    raw = true,          # without axes or grid
    camera = campixel!,  # gives a concrete bounding box in pixels
                  # so that the `vbox` gives you the right size
    width = (            # make the colorlegend longer so it looks nicer
        30,              # the width
        540              # the height
        )
    )

scatter!(p1, [135],[50],[100], markersize = 3 )
axis = p1[Axis]
axis[:names, :axisnames] = ("C", "b", "a")

scene_final = vbox(p1, cm)



Makie.save(joinpath(Base.source_dir(), "output", "parameter_correlations.png"), scene_final)
