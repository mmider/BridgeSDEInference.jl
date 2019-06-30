using Makie
using Bridge
using Bridge: increment
using StaticArrays
using Trajectories
using Colors
import Trajectories: Trajectory
using Random
Trajectory(X::SamplePath) = Trajectory(X.tt, X.yy)
simid = 2
RECORD = true
const ℝ = SVector
const parametrisation = :regular
include("../src/fitzHughNagumo.jl")
Random.seed!(10)
point(x) = Makie.Point2f0(x)
point(x, y) = Makie.Point2f0(x, y)
if simid == 1
                      # ϵ    s    γ    β    σ
P = FitzhughDiffusion(0.1, 0.0, 1.5, 0.8, 0.3)
else
P = FitzhughDiffusion(0.1, 0.0, 1.0, 0.8, 0.3)
end

Wnr = Wiener()
x0 = ℝ{2}(-0.5, -0.6)
L = @SMatrix [1. 0.]

dt = 5e-4
T = 30.0
tt = 0.0:dt:T

function simulateSegment(::S, tt, x0, P, W) where S
    WW = Bridge.samplepath(tt, zero(S))
    sample!(WW, Wnr)
    X = Trajectory(solve(Euler(), x0, WW, P))
    X, X.x[end]
end

X, _ = simulateSegment(0.0, tt, x0, P, Wnr)

#P = FitzhughDiffusion(10.0, -8.0, 15.0, 0.0, 3.0)
R = 1.5
r = range(-R, R, length=15)

limits = FRect(-R, -R, 2R, 2R)
p = Scene(resolution=(1600,1000), limits=limits)

A = vec([point(v, u) for v in r, u in r])
V = vec([point(b(NaN, ℝ{2}(v, u), P)) for v in r, u in r])
C = Bridge.viridis(last.(V))
r = range(-R, R, length=100)


arrows!(p, A, V, arrowsize=0.05, linecolor = C, arrowcolor=C,lengthscale=0.01)

lines!(p, [x for x in Point2f0.(r, r - r.^3) if x in limits])
lines!(p, [x for x in Point2f0.(r, P.γ*r .+ P.β) if x in limits])
axis = p[Axis]
axis[:names][:axisnames][] = ("V","U")
lines!(p, first.(X.x[1:2000]), last.(X.x[1:2000]))
display(p)
#savefig(p, joinpath("output", "klines.svg"))
save(joinpath("output", "klines$simid.png"), p)

ii = 5000:25000
vcol = Bridge.viridis(1:4)[2]
ucol = Bridge.viridis(1:4)[3]
p = Scene(resolution=(1600,1000))
lines!(p, X.t[ii], first.(X.x[ii]), linewidth= 2.0, color=vcol)
lines!(p, X.t[ii], last.(X.x[ii]), color=ucol)
up, down = 0.5,-0.5
lines!(p, X.t[ii], [up for i in ii], color=colorant"rgb(255,140,0)", linewidth=2.0, linestyle = :dash)
lines!(p, X.t[ii], [down for i in ii], color=colorant"rgb(255,140,0)", linewidth=2.0, linestyle = :dash)

function findCrossings(X, upLvl, downLvl)
    upSearch=true
    upCrossingTimes = Float64[0.0]
    for (x,t) in zip(X.x, X.t)
        if upSearch && x[1] > upLvl
            upSearch = false
            push!(upCrossingTimes, t)
        elseif !upSearch && x[1] < downLvl
            upSearch = true
        end
    end
    upCrossingTimes
end
c0 = findCrossings(X, up, down)
c = filter(t ->  X.t[ii[1]] < t < X.t[ii[end]], c0)
if !isempty(c)
    scatter!(p, c, [up for c in c], color=vcol)
end

axis = p[Axis]
axis[:names][:axisnames][] = ("t","V, U")
display(p);

save(joinpath("output", "trajectory$simid.png"), p)

error("STOP ")


k = 25 # tail
n = 2500 # points
pts = rand(X.x, n)
xraw = [copy(pts) for i in 1:k]
x = [Node(xraw[i]) for i in 1:k]
col = [Node(RGBA{Float32}(0.0, 0.0, 0.0, 0.0)) for i in 1:k]
c = 1
ms = 0.02
i = 1;
R = 1.5
limits = FRect(-R, -R, 2R, 2R)
p = Scene(resolution=(800,800), limits=limits, backgroundcolor = RGB{Float32}(0.04, 0.11, 0.22))
for i in randperm(k)
    scatter!(p, x[i], color = col[i], markersize = ms, #=show_axis = true,limits=limits,=#  glowwidth = 0.005, glowcolor = :white)
end
r = range(-R, 2R, length=100)
lines!(p, [x for x in Point2f0.(r, r - r.^3) if x in limits], color = RGBA{Float32}(0.5, 0.7, 1.0, 0.8))
lines!(p, [x for x in Point2f0.(r, P.γ*r .+ P.β) if x in limits], color = RGBA{Float32}(0.5, 0.7, 1.0, 0.8))
#update_cam!(p, limits)
axis = p[Axis]
axis[:grid, :linewidth] =  (1, 1)
axis[:grid, :linecolor] = (RGBA{Float32}(0.5, 0.7, 1.0, 0.5),RGBA{Float32}(0.5, 0.7, 1.0, 0.5))
axis[:names][:textsize] = (0.0,0.0)
axis[:ticks, :textcolor] = (RGBA{Float32}(0.5, 0.7, 1.0, 0.8),RGBA{Float32}(0.5, 0.7, 1.0, 0.8))

display(p)

rebirth(α, R) = x -> (rand() > α  ? x : (2rand(typeof(x)) .- 1)*R)
function update!(t, x, y, dt, P, W, jump)
    for i in eachindex(x)
        y[i] = jump(x[i])
        y[i] = y[i] + b(t, y[i], P)*dt + σ(t, y[i], P)*rand(increment(dt, W))
    end
    y
end

update!(x, y, dt, P, W, jump = indentity) = update!(NaN, x, y, dt, P, W, jump)
sleep(1)
N = 4000
jump = rebirth(0.0001, R)
for i in 1:N
    global c
    cnew = mod1(c+1, k)
    update!(x[c][], x[cnew][], dt, P, Wnr, jump)
    c = cnew
    x[c][] = x[cnew][]

    for i in 0:k-1
        f = (i + 5*(i!=0))/(k+5)
        col[mod1(c-i, k)][] = RGBA{Float32}(0.5, 0.7, 1.0, (1-sqrt(f))/3)
    end
    yield()
end

if RECORD
#convert -coalesce fitzhugh.gif fitzhugh%d.png

record(p, "output/fitzhugh$simid.gif", 1:(N÷50)) do iter
    global c
    for j in 1:5
        cnew = mod1(c+1, k)
        update!(x[c][], x[cnew][], dt, P, Wnr, jump)
        c = cnew
        x[c][] = x[cnew][]

        for i in 0:k-1
            f = (i + 5*(i!=0))/(k+5)
            col[mod1(c-i, k)][] = RGBA{Float32}(0.5, 0.7, 1.0, (1-sqrt(f))/3)
        end
    end
end
end
