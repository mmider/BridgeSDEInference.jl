using Makie
using Random
struct FitzhughNagumo{T}
    ϵ::T
    s::T
    γ::T
    β::T
end
Random.seed!(10)
point(x) = Makie.Point2f0(x)
point(x, y) = Makie.Point2f0(x, y)
const Point = Makie.Point2f0

f(x, P::FitzhughNagumo) = Point((x[1]-x[2]-x[1]^3+P.s)/P.ϵ,
                                         P.γ*x[1]-x[2] + P.β)

f(x) = f(x, P)

P = FitzhughNagumo(0.1, 0.0, 1.5, 0.8)
R = 1.5
l = (32, 32)
limits = FRect(-R, -R, 2R, 2R)
r = (range(limits.origin[1], limits.origin[1] + limits.widths[1], length=l[1]+1),
 range(limits.origin[2], limits.origin[2] + limits.widths[2], length=l[2]+1)
)
#p = Scene(resolution=(1600,1000), limits=limits)

mask = trues(l...)
arrows = Tuple{Point,Point}[]

for c in Random.shuffle(CartesianIndices(mask))

    i0, j0 = Tuple(c)
    x0 = Point(first(r[1]) + (i0 - 0.5)*step(r[1]),
              first(r[2]) + (j0 - 0.5)*step(r[1]))
    dt = 0.01

    if mask[c]
        push!(arrows, (x0, f(x0, P)/norm(f(x0, P))))

        mask[c] == false
        for d in (-1,1)
            x = x0
            xx = [x]
            i0, j0 = Tuple(c)
            while x in limits #&& length(xx) < 2500
                x = x + d*dt*f(x, P)/norm(f(x, P))
                if !(x in limits)
                    break
                end
                i = searchsortedlast(r[1], x[1])
                j = searchsortedlast(r[2], x[2])
                if (i,j) != (i0, j0)
                    if !mask[i,j]
                        break
                    end
                    mask[i,j] = false
                    i0, j0 = i, j
                end
                push!(xx, x)
            end
            lines!(p, xx, linewidth=2.0, color=:orange)
        end
    end

end
#arrows!(p, first.(arrows), last.(arrows), color=:orange, lengthscale=0.0, arrowsize=0.05)
display(p)
