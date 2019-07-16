# Silly example with slider
using Makie
sr, or = textslider(10:1000, "resolution", start = 20);

parent = Scene(resolution = (1000, 500))

t = lift(or) do resolution
  LinRange(0.0, 100.0, resolution)
end
y = lift(t) do t; sin.(t); end

pl = lines(t, y)
scene = hbox(sr, pl, parent=parent)
