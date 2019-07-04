using Makie, Makie.AbstractPlotting
using Makie.AbstractPlotting: textslider

using StaticArrays

using Random
using Distributions
using Colors

RECORD = false
R = 1.5 # ~size of the interesting region around the origin
dt = 0.01
sq_dt = sqrt(dt)


# One slider for each parameter
s1, ϵ = textslider(0.001f0:0.001f0:3.0f0, "ϵ", start = 0.1)
s2, s = textslider(-5.0f0:0.001f0:5.0f0, "s", start = 0.0)
s3, γ = textslider(-3.0f0:0.001f0:3.0f0, "γ", start = 1.5)
s4, β = textslider(-3.0f0:0.001f0:3.0f0, "β", start = 0.8)
s5, σ₁ = textslider(0.0f0:0.001f0:1.0f0, "σ₁", start = 0.0)
s6, σ₂ = textslider(0.0f0:0.001f0:3.0f0, "σ₂", start = 0.3)

# Define the dynamics of the process
function drift(x::Point2f0, ϵ=0.1, s=0.0, γ=1.5, β=0.8)
    Point2f0((x[1] - x[2] - x[1]^3 + s)/ϵ, γ*x[1]-x[2] + β)
end

function vola(x::Point2f0, σ₁=0.0, σ₂=0.3)
    Point2f0(σ₁, σ₂)
end

drift_param = (ϵ, s, γ, β)
vola_param = (σ₁, σ₂)

rebirth(α, R) = x -> (rand() > α  ? x : (2rand(typeof(x)) .- 1)*R)
function update!(t, x, y, jump)
    for i in eachindex(x)
        y[i] = jump(x[i])
        y[i] = (y[i] + drift(y[i], to_value.(drift_param)...)*dt
                     + sq_dt * vola(y[i], to_value.(vola_param)...).* Point2f0(randn(2)))
    end
    y
end

# Define traces zapping around the space & initialise them
trace_len = 250
num_pts = 2500
starting_pts = [Point2f0(2*R*rand(2).-R) for i in 1:num_pts]
xraw = [copy(starting_pts) for i in 1:trace_len]
x = [Node(xraw[i]) for i in 1:trace_len]
col = [Node(RGBA{Float32}(0.0, 0.0, 0.0, 0.0)) for i in 1:trace_len]
c = 1
ms = 0.02
#i = 1;

limits = FRect(-R, -R, 2R, 2R)
particles_canvas = Scene(resolution=(800,800), limits=limits, backgroundcolor = RGB{Float32}(0.04, 0.11, 0.22))
for i in randperm(trace_len)
    scatter!(particles_canvas, x[i], color = col[i], markersize = ms, #=show_axis = true,limits=limits,=#  glowwidth = 0.005, glowcolor = :white)
end
#update_cam!(p, limits)
axis = particles_canvas[Axis]
axis[:grid, :linewidth] =  (1, 1)
axis[:grid, :linecolor] = (RGBA{Float32}(0.5, 0.7, 1.0, 0.5),RGBA{Float32}(0.5, 0.7, 1.0, 0.5))
axis[:names][:textsize] = (0.0,0.0)
axis[:ticks, :textcolor] = (RGBA{Float32}(0.5, 0.7, 1.0, 0.8),RGBA{Float32}(0.5, 0.7, 1.0, 0.8))

parent = Scene(resolution = (1000, 800))
vbox(hbox(s1, s2, s3, s4, s5, s6), particles_canvas, parent = parent)
display(parent)



update!(x, y, jump = indentity) = update!(NaN, x, y, jump)
sleep(1)
N = 4000
jump = rebirth(0.0001, R)
for i in 1:N
    global c
    cnew = mod1(c+1, trace_len)
    update!(x[c][], x[cnew][], jump)
    c = cnew
    x[c][] = x[cnew][]

    for i in 0:trace_len-1
        f = (i + 5*(i!=0))/(trace_len+5)
        col[mod1(c-i, trace_len)][] = RGBA{Float32}(0.5, 0.7, 1.0, (1-sqrt(f))/3)
    end
    sleep(1e-10)
end

if RECORD
record(p, "output/fitzhugh.mp4", 1:(N÷10)) do i
    global c
    cnew = mod1(c+1, trace_len)
    update!(x[c][], x[cnew][], jump)
    c = cnew
    x[c][] = x[cnew][]

    for i in 0:trace_len-1
        f = (i + 5*(i!=0))/(trace_len+5)
        col[mod1(c-i, trace_len)][] = RGBA{Float32}(0.5, 0.7, 1.0, (1-sqrt(f))/3)
    end
    yield()
end
end
