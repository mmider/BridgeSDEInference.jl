using Makie
using Bridge: increment
using StaticArrays
using Trajectories
using Colors
Trajectory(X::SamplePath) = Trajectory(X.tt, X.yy)

const parametrisation = :regular
include("../src/fitzHughNagumo.jl")

P = FitzhughDiffusion(10.0, -8.0, 15.0, 0.0, 3.0)
Wnr = Wiener()
x0 = ℝ{2}(-0.5, -0.6)
if parametrisation == :simpleAlter
    x0 = regularToAlter(x0, P.ϵ, 0.0)
elseif parametrisation == :simpleConjug
    x0 = regularToConjug(x0, P.ϵ, 0.0)
end
L = @SMatrix [1. 0.]

dt = 1e-3
T = 30.0
tt = 0.0:dt:T

function simulateSegment(::S, tt, x0, P, W) where S
    WW = Bridge.samplepath(tt, zero(S))
    sample!(WW, Wnr)
    X = Trajectory(solve(Euler(), x0, WW, P))
    X, X.x[end]
end

X, _ = simulateSegment(0.0, tt, x0, P, Wnr)

k = 50


xraw = [X.x[1:100:end] for i in 1:k]
x = [Node(xraw[i]) for i in 1:k]
col = [Node(RGB{Float32}(0.0, 0.0, 0.0)) for i in 1:k]
c = 1

#h = on(x) do val
#   println("Got an update: ", typeof(val))
#end

p = scatter(x[1], color = col[1])
for i in 2:k
    scatter!(p, x[i], color = col[i])
end
display(p)

function update!(t, x, y, dt, P, W)
    for i in eachindex(x)
        y[i] = x[i] + b(t, x[i], P)*dt + σ(t, x[i], P)*rand(increment(dt, W))
    end
    y
end

update!(x, y, dt, P, W) = update!(NaN, x, y, dt, P, W)
sleep(1)
for i in 1:3000
    global c
    cnew = mod1(c+1, k)
    update!(x[c][], x[cnew][], dt, P, Wnr)
    c = cnew
    x[c][] = x[cnew][]
    #f = mod1(i, 3k)/3k
    #col[mod1(c, k)][] = RGB{Float32}(f, f, 1-f)
    for i in 0:k-1
        f = (i)/k
        col[mod1(c-i, k)][] = RGB{Float32}(f, f, 1.0)
    end
    sleep(0.000001)
end
