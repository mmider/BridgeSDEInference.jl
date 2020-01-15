using Bridge
using Random
using DataFrames
using CSV
using BridgeSDEInference
using RCall

# Home directory ".../BridgeSDEInference"
HOME_DIR = "../../BridgeSDEInference"
OUT_DIR = joinpath(HOME_DIR, "output")
mkpath(OUT_DIR)
FILENAME_OUT = joinpath(OUT_DIR, "FS_testdata.csv")

### parameters as in FS
FT = 70.; VB = 20.; PS = VE = 15.; HE = 0.4; DT = 2.4 # Favetto-Samson
DT = 1. # out choice
P = FS(FT/(1-HE), FT/(VB*(1-HE)), PS/(VB*(1-HE)), 1.5* PS/VE, sqrt(2),0.2)

# starting point under :regular parametrisation

x0 = ℝ{2}(0.0, 0.0)

dt = 0.000001
T = 10.0
tt = 0.0:dt:T

Random.seed!(4)
XX, _ = simulate_segment(ℝ{2}(0.0, 0.0), x0, P, tt)


num_obs = 60
skip = div(length(tt), num_obs)
Time = collect(tt)[1:skip:end]
#observation x[2]- x[3]
df = DataFrame(time=Time, x1=[0.5*x[1] + 0.5*x[2] for x in XX.yy[1:skip:end]])
CSV.write(FILENAME_OUT, df)

df_full = DataFrame(time=Time, x1=[x[1] for x in XX.yy[1:skip:end]], x2=[x[2] for x in XX.yy[1:skip:end]])
@rput df_full
@rput df
R"""
library(ggplot2)
library(tidyverse)
df_full2 <- df_full %>% gather(key='component',value='value', x1,x2)
 ggplot() + geom_path(data=df_full2, aes(x=time,y=value,colour=component)) + geom_point(data=df_full2, aes(x=time,y=value,colour=component)) +
   geom_point(data=df, aes(x=time,y=x1))

"""


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
