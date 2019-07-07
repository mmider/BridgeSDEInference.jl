using Makie, Makie.AbstractPlotting
using Makie.AbstractPlotting: textslider
using Random
using Colors
using LinearAlgebra

struct FitzhughNagumoDiff{T}
    ϵ::T
    γ::T
    β::T
    σ₁::T
    σ₂::T
end

R = 1.5f0 # size of the interesting region around the origin
dt = 0.005 # time step
sq_dt = sqrt(dt)

trace_len = 40 # number of points in the comets' tails
num_pts = 250 # number of comets

# comets die and are reborn at random places
rebirth(α, R) = x -> (rand() > α  ? x : (2rand(typeof(x)) .- 1)*R)
jump = rebirth(0.0001, R) # location of point creation

# starting points of the comets as if they were are reborn

starting_pts = [rebirth(1.0, R)(zero(Point2f0)) for i in 1:num_pts]

T = 8.0 # time frame for the rightmost panel
graph_time = collect(0:dt:T)
graph_starttime = 0.0

N = 4000 # run for N steps


# One slider for each parameter
s1, σ₂ = textslider(0.0f0:0.001f0:3.0f0, "σ₂", start = 0.3)
s2, σ₁ = textslider(0.0f0:0.001f0:1.0f0, "σ₁", start = 0.0)
s3, β = textslider(-3.0f0:0.001f0:3.0f0, "β", start = 0.8)
s4, γ = textslider(-3.0f0:0.001f0:3.0f0, "γ", start = 1.5)
s5, ϵ = textslider(0.001f0:0.001f0:3.0f0, "ϵ", start = 0.1)

# Model containing parameters governing drift and diffusion
diffusion = lift(ϵ, γ, β, σ₁, σ₂) do ϵ, γ, β, σ₁, σ₂
    FitzhughNagumoDiff(ϵ, γ, β, σ₁, σ₂)
end

# Define the dynamics of the process

# directional component of the movement of the planets
function drift(x::Point2f0, P::FitzhughNagumoDiff)
    Point2f0((x[1] - x[2] - x[1]^3)/P.ϵ, P.γ*x[1] - x[2] + P.β)
end

# volality is the amount of stochastic movement in x and y direction
function vola(x::Point2f0, P::FitzhughNagumoDiff)
    Diagonal(Point2f0(P.σ₁, P.σ₂))
end

# random update for the comets
# acts on vectors of points
function update!(t, xin, xout, diffusion, jump)
    for i in eachindex(xin)
        xout[i] = jump(xin[i])
        xout[i] = (xout[i] + drift(xout[i], diffusion)*dt
                     + sq_dt * vola(xout[i], diffusion) * Point2f0(randn(2)))
    end
    xout
end
update!(x, y, P, jump = identity) = update!(NaN, x, y, P, jump) # short version for now jumps

function update(x, P)
    (x + drift(x, P)*dt
       + sq_dt * vola(k, P) * randn(2))
end


# Define comets and their tails

# how fast does the tail fade off?
fdecay(i, len) = (i + 5*(i!=0))/(len+5)

# circular array of vectors of points,
xraw = [copy(starting_pts) for i in 1:trace_len]
x = [Node(xraw[i]) for i in 1:trace_len]
# each vector contains points of the same age/color
col = [Node(RGBA{Float32}(0.0, 0.0, 0.0, 0.0)) for i in 1:trace_len]

# seperate color vector for special yellow comet tracking the right plot
col1 = [RGBA{Float32}(0.0, 0.0, 0.0, 0.0) for i in graph_time]

# compute vector of colors. At first, `c = end` is the newest vertex
c = trace_len
for i in 0:trace_len-1
    f = fdecay(i, trace_len)
    col[mod1(c-i, trace_len)][] = RGBA{Float32}(0.5, 0.7, 1.0, (1-sqrt(f))/3)
end
trace_len1 = length(graph_time)÷2
c = length(graph_time)
for i in 0:trace_len1-1
        f = fdecay(i, trace_len1)
        col1[mod1(c-i, length(graph_time))] = RGBA{Float32}(1.0, 0.7, 0.5, (1-sqrt(f))/2)
end

ms = 0.03

limits = FRect(-R, -R, 2R, 2R)
bg_col = RGB{Float32}(0.04, 0.11, 0.22)
particles_canvas = Scene(resolution=(800,800), limits=limits, backgroundcolor = bg_col)
r = range(-R, 2R, length=100)
pline = lift(γ, β) do γ, β
  [x for x in Point2f0.(r, γ*r .+ β) if x in limits]
end

# null klines
lines!(particles_canvas, [x for x in Point2f0.(r, r - r.^3) if x in limits], color = RGBA{Float32}(0.5, 0.7, 1.0, 0.5))
lines!(particles_canvas, pline, color = RGBA{Float32}(0.5, 0.7, 1.0, 0.5))


for i in randperm(trace_len)
    scatter!(particles_canvas, x[i], color = col[i], markersize = ms, #=show_axis = true,limits=limits,=#  glowwidth = 0.005, glowcolor = :white)
end


# Define graph tracing history of each coordinate of a single particle

graph_ts = graph_starttime .+ graph_time
graph_ys = #nodes of vectors for first and second coordinate
Node([starting_pts[1][1] for t in graph_time]),
Node([starting_pts[1][2] for t in graph_time])


lines!(particles_canvas, graph_ys..., color = col1, linewidth=1.5)

limits2 = FRect(graph_ts[1], -R, graph_ts[end]-graph_ts[1], 2R)

graph_canvas = Scene(resolution=(400,800), #=limits=limits2 ,=# backgroundcolor = bg_col)
plotted_line1 = lines!(graph_canvas, graph_ts, graph_ys[1], color=:white)
plotted_line1 = plotted_line1[end]
plotted_line2 = lines!(graph_canvas, graph_ts, graph_ys[2], color=:goldenrod)
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
vbox(hbox(s1, s2, s3, s4, s5), particles_canvas, graph_canvas, parent = parent)
display(parent)





sleep(0.5)

c = 1 # circular counter keeping track of location of newest/most recent
      # comet position

for i in 1:N
#    record(parent, "output/fitzhugh.mp4", 1:(N÷10)) do i
    global c
    global graph_starttime
    cnew = mod1(c+1, trace_len) # update index
    update!(x[c][], x[cnew][], diffusion[], jump) # update planet position
                    # overwrites vector of oldest tail points
    c = cnew
    x[c][] = x[cnew][] # trigger repaint

    for i in 0:trace_len-1 # older planet location is dimmed down
        f = (i + 5*(i!=0))/(trace_len+5)
        col[mod1(c-i, trace_len)][] = RGBA{Float32}(0.5, 0.7, 1.0, (1-sqrt(f))/3)
    end

    # also track the location of the first planet in rightmost panel
    graph_ys[1][][1:end-1] = graph_ys[1][][2:end]
    graph_ys[2][][1:end-1] = graph_ys[2][][2:end]
    p1 = x[c][][1]
    for d in 1:2
        graph_ys[d][] = [graph_ys[d][][1:end-1]; p1[d]]
    end

    # update time axis
    graph_starttime += dt
    plotted_line1[1] = graph_starttime .+ graph_time
    plotted_line2[1] = graph_starttime .+ graph_time

    AbstractPlotting.update_limits!(graph_canvas)
    AbstractPlotting.update!(graph_canvas)

    #yield()
    sleep(0.001) 
end
