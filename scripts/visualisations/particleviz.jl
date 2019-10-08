using Makie
using Bridge
using Bridge: increment
using StaticArrays
using Trajectories
using Colors
import Trajectories: Trajectory
using Random
Trajectory(X::SamplePath) = Trajectory(X.tt, X.yy)
Wnr = Wiener()
P = P̂
R = 𝕏(1.5,6.0)
rebirth(α, R) = x -> (rand() > α  ? x : (2rand(typeof(x)) .- 1).*R)

k = 250
n = 250
pts = [rebirth(1.0, R)(x0[1]) for i in 1:n]
xraw = [copy(pts) for i in 1:k]
x = [Node(xraw[i]) for i in 1:k]
col = [Node(RGBA{Float32}(0.0, 0.0, 0.0, 0.0)) for i in 1:k]
c = 1
ms = 0.08
i = 1;

limits = FRect(-R[1], -R[2], 2R[1], 2R[2])
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
r = range(-R[1], 2R[1], length=100)

display(p)


function update!(t, x, y, dt, P, W, jump)
    for i in eachindex(x)
        y[i] = jump(x[i])
        y[i] = y[i] + BSI.b(t, y[i], P)*dt + BSI.σ(t, y[i], P)*rand(increment(dt, W))
    end
    y
end
dtf = dt
update!(x, y, dt, P, W, jump = indentity) = update!(NaN, x, y, dt, P, W, jump)
sleep(1)
N = 4000
jump = rebirth(0.0001, R)
for i in 1:N
    global c
    cnew = mod1(c+1, k)
    update!(x[c][], x[cnew][], dtf, P, Wnr, jump)
    c = cnew
    x[c][] = x[cnew][]

    for i in 0:k-1
        f = (i + 5*(i!=0))/(k+5)
        col[mod1(c-i, k)][] = RGBA{Float32}(0.5, 0.7, 1.0, (1-sqrt(f))/3)
    end
    sleep(1e-10)
end
