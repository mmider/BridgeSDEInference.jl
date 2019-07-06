using Makie, Makie.AbstractPlotting
using Makie.AbstractPlotting: textslider
using Random
using Colors

RECORD = false
R = 1.5 # ~size of the interesting region around the origin
dt = 0.01
sq_dt = sqrt(dt)


# One slider for each parameter
s1, σ₂ = textslider(0.0f0:0.001f0:3.0f0, "σ₂", start = 0.3)
s2, σ₁ = textslider(0.0f0:0.001f0:1.0f0, "σ₁", start = 0.0)
s3, β = textslider(-3.0f0:0.001f0:3.0f0, "β", start = 0.8)
s4, γ = textslider(-3.0f0:0.001f0:3.0f0, "γ", start = 1.5)
s5, s = textslider(-5.0f0:0.001f0:5.0f0, "s", start = 0.0)
s6, ϵ = textslider(0.001f0:0.001f0:3.0f0, "ϵ", start = 0.1)

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
update!(x, y, jump = indentity) = update!(NaN, x, y, jump)

function update(x)
    k = Point2f0(x)
    (k + drift(k, to_value.(drift_param)...)*dt
       + sq_dt * vola(k, to_value.(vola_param)...).* randn(2))
end


# Define traces zapping around the space
trace_len = 25
num_pts = 250
starting_pts = [Point2f0(2*R*rand(2).-R) for i in 1:num_pts]
xraw = [copy(starting_pts) for i in 1:trace_len]
x = [Node(xraw[i]) for i in 1:trace_len]
col = [Node(RGBA{Float32}(0.0, 0.0, 0.0, 0.0)) for i in 1:trace_len]
ms = 0.02

limits = FRect(-R, -R, 2R, 2R)
bg_col = RGB{Float32}(0.04, 0.11, 0.22)
particles_canvas = Scene(resolution=(800,800), limits=limits, backgroundcolor = bg_col)
r = range(-R, 2R, length=100)
pline = lift(γ, β) do γ, β
  [x for x in Point2f0.(r, γ*r .+ β) if x in limits]
end

lines!(particles_canvas, [x for x in Point2f0.(r, r - r.^3) if x in limits], color = RGBA{Float32}(0.5, 0.7, 1.0, 0.8))
lines!(particles_canvas, pline, color = RGBA{Float32}(0.5, 0.7, 1.0, 0.8))


for i in randperm(trace_len)
    scatter!(particles_canvas, x[i], color = col[i], markersize = ms, #=show_axis = true,limits=limits,=#  glowwidth = 0.005, glowcolor = :white)
end


# Define graph tracing history of each coordinate of a single particle
T = 8.0
frame_time = collect(0:dt:T)
frame_start = 0.0
frame_ys = zeros((2, length(frame_time)))
for i in 1:length(frame_time)-1
    frame_ys[:,i+1] = update(frame_ys[:,i])
end


graph_canvas = Scene(resolution=(400,800), backgroundcolor = bg_col)
plotted_line1 = lines!(graph_canvas, frame_start .+ frame_time, frame_ys[1,:], color=:white)
plotted_line1 = plotted_line1[end]
plotted_line2 = lines!(graph_canvas, frame_start .+ frame_time, frame_ys[2,:], color=:goldenrod)
plotted_line2 = plotted_line2[end]

lineplots = [plotted_line1, plotted_line2]

# Updates colors
for axis in [particles_canvas[Axis], graph_canvas[Axis]]
    axis[:grid, :linewidth] =  (1, 1)
    axis[:grid, :linecolor] = (RGBA{Float32}(0.5, 0.7, 1.0, 0.5),RGBA{Float32}(0.5, 0.7, 1.0, 0.5))
    axis[:names][:textsize] = (0.0,0.0)
    axis[:ticks, :textcolor] = (RGBA{Float32}(0.5, 0.7, 1.0, 0.8),RGBA{Float32}(0.5, 0.7, 1.0, 0.8))
end


# Define combined canvas
parent = Scene(resolution = (1400, 800))
vbox(hbox(s1, s2, s3, s4, s5, s6), particles_canvas, graph_canvas, parent = parent)
display(parent)





sleep(1)
N = 4000
jump = rebirth(0.0001, R)
c = 1

for i in 1:N
    global c
    global frame_start
    cnew = mod1(c+1, trace_len)
    update!(x[c][], x[cnew][], jump)
    c = cnew
    x[c][] = x[cnew][]

    for i in 0:trace_len-1
        f = (i + 5*(i!=0))/(trace_len+5)
        col[mod1(c-i, trace_len)][] = RGBA{Float32}(0.5, 0.7, 1.0, (1-sqrt(f))/3)
    end

    frame_ys[:,1:end-1] .= frame_ys[:,2:end]
    frame_ys[:,end] = update(frame_ys[:,end-1])
    for (l,j) in zip(lineplots, [1,2])
        l[1] = frame_start .+ frame_time
        l[2] = frame_ys[j,:]
    end
    AbstractPlotting.update!(graph_canvas)

    sleep(1e-10)
end

if RECORD
record(parent, "output/fitzhugh.mp4", 1:(N÷10)) do i
    global c
    cnew = mod1(c+1, trace_len)
    update!(x[c][], x[cnew][], jump)
    c = cnew
    x[c][] = x[cnew][]

    for i in 0:trace_len-1
        f = (i + 5*(i!=0))/(trace_len+5)
        col[mod1(c-i, trace_len)][] = RGBA{Float32}(0.5, 0.7, 1.0, (1-sqrt(f))/3)
    end

    frame_ys[:,1:end-1] .= frame_ys[:,2:end]
    frame_ys[:,end] = update(frame_ys[:,end-1])
    for (l,j) in zip(lineplots, [1,2])
        l[1] = frame_start .+ frame_time
        l[2] = frame_ys[j,:]
    end
    AbstractPlotting.update!(graph_canvas)

    yield()
end
end
