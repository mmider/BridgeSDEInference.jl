using Bridge
using Random
using DataFrames
using CSV
using BridgeSDEInference

# Home directory ".../BridgeSDEInference"
HOME_DIR = "../../BridgeSDEInference"
OUT_DIR = joinpath(HOME_DIR, "output")
mkpath(OUT_DIR)
FILENAME_OUT = joinpath(OUT_DIR, "jr_path_part_obs_3n.csv")

### parameters as Table 1  of
# https://mathematical-neuroscience.springeropen.com/articles/10.1186/s13408-017-0046-4
P = JRNeuralDiffusion3n(3.25, 100.0, 22.0, 50.0 , 135.0, 5.0, 6.0, 0.56, 0.0, 220.0, 0.0, 0.01, 2000.0, 1.0)
# starting point under :regular parametrisation

x0 = ℝ{6}(0.08, 18, 15, -0.5, 0, 0)

dt = 0.000001
T = 2.0
tt = 0.0:dt:T

Random.seed!(4)
XX, _ = simulate_segment(ℝ{3}(0.0, 0.0, 0.0), x0, P, tt)


num_obs = 256*5
skip = div(length(tt), num_obs)
Time = collect(tt)[1:skip:end]
#observation x[2]- x[3]
df = DataFrame(time=Time, x1=[x[2] - x[3] for x in XX.yy[1:skip:end]])
CSV.write(FILENAME_OUT, df)


error("STOP HERE")
using Makie

p1 = Makie.scatter(df.time, df.x1,  markersize = 0.1)
axis = p1[Axis]
axis[:names][:axisnames] = ("t", "X_2 - X_3")
Makie.lines!(p1, XX.tt, [XX.yy[i][2] - XX.yy[i][3] for i in 1:length(XX.yy)], color = (:blue, 0.7))
#Makie.save("2_minus_3.png", a)

p2 = Makie.lines(XX.tt, [XX.yy[i][1] for i in 1:length(XX.yy)])
#Makie.save("1_Latent.png", a)
axis = p2[Axis]
axis[:names][:axisnames] = ("t", "X_1")

p3 = Makie.lines(XX.tt, [XX.yy[i][4] for i in 1:length(XX.yy)])
axis = p3[Axis]
axis[:names][:axisnames] = ("t", "X_4")
#Makie.save("4_latent.png", b)

p4 = Makie.lines(XX.tt, [XX.yy[i][5] for i in 1:length(XX.yy)])
axis = p4[Axis]
axis[:names][:axisnames] = ("t", "X_5")
#Makie.save("5_latent_5seconds.png", b)

p5 = Makie.lines(XX.tt, [XX.yy[i][6] for i in 1:length(XX.yy)])
axis = p5[Axis]
axis[:names][:axisnames] = ("t", "X_6")

pscene = hbox(
    vbox(p3, p5),
    p4,
    vbox(p1, p2)
)

Makie.save("./assets/simulated_data.png", pscene)
