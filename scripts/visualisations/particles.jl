using Makie
using Bridge
using Bridge: increment
using StaticArrays
using Trajectories
using Colors
import Trajectories: Trajectory
using Random
Trajectory(X::SamplePath) = Trajectory(X.tt, X.yy)
RECORD = false
const ℝ = SVector
const parametrisation = :regular
include("../src/fitzHughNagumo.jl")
P = FitzhughDiffusion(0.1, 0.0, 1.5, 0.8, 0.3)
#P = FitzhughDiffusion(10.0, -8.0, 15.0, 0.0, 3.0)
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

k = 250
n = 2500
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
#update_cam!(p, limits)
axis = p[Axis]
axis[:grid, :linewidth] =  (1, 1)
axis[:grid, :linecolor] = (RGBA{Float32}(0.5, 0.7, 1.0, 0.5),RGBA{Float32}(0.5, 0.7, 1.0, 0.5))
axis[:names][:textsize] = (0.0,0.0)
axis[:ticks, :textcolor] = (RGBA{Float32}(0.5, 0.7, 1.0, 0.8),RGBA{Float32}(0.5, 0.7, 1.0, 0.8))
r = range(-R, 2R, length=100)
lines!(p, [x for x in Point2f0.(r, r - r.^3) if x in limits], color = RGBA{Float32}(0.5, 0.7, 1.0, 0.8))
lines!(p, [x for x in Point2f0.(r, P.γ*r .+ P.β) if x in limits], color = RGBA{Float32}(0.5, 0.7, 1.0, 0.8))

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
    sleep(1e-10)
end

if RECORD
record(p, "output/fitzhugh.mp4", 1:(N÷10)) do i
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
end
