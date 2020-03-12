using Bridge
using Random
using DataFrames
using CSV
using BridgeSDEInference

# Home directory ".../BridgeSDEInference"
OUT_DIR ="./output"
mkpath(OUT_DIR)
FILENAME_OUT1 = joinpath(OUT_DIR, "ecofin_path_interest_rates.csv")
FILENAME_OUT2 = joinpath(OUT_DIR, "ecofin_path_indexes.csv")
### parameters as Table 1  of
# https://mathematical-neuroscience.springeropen.com/articles/10.1186/s13408-017-0046-4
ρ = 0.03
δ = 0.05
γ = 0.1
κ = 0.2
η = 0.01
σ = 0.02
P = EcoFinEq(ρ, δ, γ, κ, η, σ)
# starting point under :regular parametrisation
x0 = ℝ{3}(log(P.ρ/P.γ), 0.0, P.γ - P.δ - P.σ^2)

dt = 0.1/360
T = 10.0
tt = 0.0:dt:T

Random.seed!(4)
XX, _ = simulate_segment(ℝ{2}(0.0, 0.0), x0, P, tt)

num_daily_obs = Int(T)*360
skip = div(length(tt), num_daily_obs)
Time = collect(tt)[1:skip:end]
#observation x[2]- x[3]
XX.yy[1:skip:end]
df_interest_rates = DataFrame(time=Time, x3=[ x[3] for x in XX.yy[1:skip:end] ])

num_quarterly_obs = 25*4
skip1 = Int(div(length(tt), num_quarterly_obs))
Time1 = collect(tt)[1:skip1:end]
df_quarterly_indexes = DataFrame(time=Time1, x1 = [ x[1] for x in XX.yy[1:skip1:end] ], x2=[ x[2] for x in XX.yy[1:skip1:end] ] )

CSV.write(FILENAME_OUT1, df_interest_rates)
CSV.write(FILENAME_OUT2, df_quarterly_indexes)

error("STOP SIMULATION HERE")

##Plot the simulated data
using Makie
p1 = lines(XX.tt, exp.([XX.yy[i][1] for i in 1:length(XX)]))
p2 = lines(XX.tt, exp.([XX.yy[i][2] for i in 1:length(XX)]), color = :red)
p3 = lines(XX.tt, [XX.yy[i][3] for i in 1:length(XX)], color = :green)
p_final = hbox(p1,p2,p3)
resize!(p_final, 5000,3000)
save("output/ecofin_equilibrium_model.png", p_final)
