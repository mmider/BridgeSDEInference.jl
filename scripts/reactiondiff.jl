using Makie
using Random
using Test
using LinearAlgebra
using StaticArrays
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
S = 9
function graphparam(S = S)
    global l = (1<<S, 1<<S)

    global limits = FRect(-R, -R, 2R, 2R)
    global r = (range(limits.origin[1], limits.origin[1] + limits.widths[1], length=l[1]+1),
 range(limits.origin[2], limits.origin[2] + limits.widths[2], length=l[2]+1)
)
end
graphparam()

function saveuv()
    global  U2 = copy(U)
    global V2 = copy(V)
end
function double()
    global U = repeat(U2, inner=(2,2))
    global V = repeat(V2, inner=(2,2))
    Vshow[] = V
end

U = 1ones(l...)
V = 0.05rand!(zeros(l...))

for iter in 1:round(Int,0.0004*prod(size(V)))
    i = rand(6:(l[1]-6))
    j = rand(6:(l[2]-6))
    V[i:(i+0),j:(j+1)] .= 1.0 #rand(2,2)
end

for i in 6:32:l[1]-6
    for j in 6:32:l[2]-6
    #    V[i:(i+1),j:(j+1)] .= 1.0
    end
end

i, j = l.>> 1
#U[(i-5):(i+6),(j-5):(j+6)] .= 0.5 .+ 0.01*rand(12,12)
#V[(i-5):(i+6),(j-5):(j+6)] .= 0.25 .+ 0.01*rand(12,12)

#end

#https://www.algosome.com/articles/reaction-diffusion-gray-scott.html

L3() = @SMatrix [0.05 0.2 0.05
     0.2  -1.0 0.2
     0.05 0.2 0.05]

L5() = (1/16)*@SMatrix [0 0 1 0 0; 0 1 2 1 0; 1 2 -16 2 1;  0 1 2 1 0; 0 0 1 0 0 ]

# https://en.wikipedia.org/wiki/Discrete_Laplace_operator
L2d(γ = 1/3) = (1-γ)*@SMatrix([0 1 0; 1 -4 1; 0 1 0]) + γ*@SMatrix([1/2  0 1/2; 0 -2 0; 1/2 0 1/2])

#f(x,y) = cos(atan(x,y))^2 - y^2/(x^2+y^2)

A(a1, a2, θ) = ((a1*cos(θ))^2 + (a2*sin(θ))^2), (a2^2 - a1^2)*sin(θ)*cos(θ), ((a2*cos(θ))^2 + (a1*sin(θ))^2)
function Lani(a11, a12, a22)
    [ -a12 2a22 a12
      2a11 -4*(a11 + a22)  2a11
      a12  2a22 -a12]/(4*(a11 + a22))
end
# anisotropic
function Lani(a1, a2, x, y)
    h = (x^2+y^2)
    a11 = ((a1*y)^2 + (a2*x)^2)/h
    a12 = (a2^2 - a1^2)*(x*y)/h
    a22 = ((a1*x)^2 + (a2*y)^2)/h

    (@SMatrix [ -a12 2a22 a12
      2a11 -4*(a11 + a22)  2a11
      a12  2a22 -a12])/(4*(a11 + a22))
end
@test norm(Lani(0.1, 1, 0.5, 1) - Lani(A(0.1, 1, atan(0.5, 1))...)) < 0.01
#https://github.com/rodrigosetti/reaction-diffusion/blob/master/src/reactdiffuse.c
#http://www.karlsims.com/rd.html

nlz(A) = A/maximum(abs.(A))

L = L2d()

vary = false
# normal
 F, K = .055, .062

# mitosis
 F, K =  .0367, .0649

# coral
#F, K = .0545, .062

# mazes
#F, K = .029, .057

# worms
#F, K = .078, .061

# spots
#F, K = 0.0322, 0.0556

# connected spots
#F, K = 0.029, 0.0555


# rings
#F, K = 0.07, 0.062


# aniso moving
#F, K = 0.037, 0.058

# aniso mitosis
#F, K = 0.021, 0.069

# aniso interrupt
#F, K = 0.022, 0.066

# aniso waves
#F, K = 0.024, 0.052

# xmax, ycenter

#F, K, vary = 0.12, 0.06, true
#F, K, vary = 0.12, 0.06, true
if !vary
# hand picked
#F, K = 0.023, 0.054
#F = 0.013; K = 0.044 # pulsating
#F = 0.008; K = 0.036 # instable
#F = 0.035; K = 0.054 # even but to smeared in the boundary
#F = 0.026; K = 0.056 # fingerprint
#vary = false
#lines!(p, collect(r[1]), -1.5 .+ 1.5*max.(1 .- (0.75*r[1]).^2, 0), color=:white)
#lines!(p, collect(r[1]), -1 .+ 1.5*max.(1 .- (0.5*r[1]).^2, 0), color=:white)

#F = 0.015; K = 0.06 # curve mimosis

#F = 0.02; K = 0.066 # directed mimosis
#F = 0.02; K = 0.057 # tail fighter
#F = 0.008; K = 0.052 # dying fighters
#F = 0.012; K = 0.052 # surviving fighters

#F = 0.101; K = 0.056; # slow eyes

F = 0.071; K = 0.0605 # long lines 0.061 boundary

end #!vary

du = 1.0
dv = 0.5 # 0.5 # increased for coarser image
dt = 1.0

mirror(i) = abs(i-1) + 1
mirror(i, n) =  n - abs(n - mirror(i))


function laplace(U, L, c)
    s = 0.0
    m, n = size(U)
    d = (size(L, 1)>>1)
    @inbounds for i in -d:d
        c1 = mirror(c[1]+i, m)
        @inbounds for j in -d:d
            s += @inbounds L[1+d + i, 1+d + j]*U[c1, mirror(c[2]+j, n)]
        end
    end
    s
end

function toFK(F, K, x; reparam=true)
    F1 = F*(1.5+x[1])/3
    y = x[2]
    if reparam
        y = -1 + x[2] + 1.5*max.(1 .- (0.75*x[1]).^2, 0)
    end
    K1 = K + y/100
    #K1 = K + x[2]/50
    F1, K1
end
#http://www.cs.cmu.edu/~jkh/462_s07/reaction_diffusion.pdf
pt(r, c) = point(r[1][c[1]], r[2][c[2]])
idx(r, x) = searchsortedlast(r[1], x[1]), searchsortedlast(r[2], x[2])
function react!(U, V, f, P, (F, K, du, dv, dt); iters=18800, homog=false, vary=false, anneal=false, scale=false, reparam=false)
    Uᵒ = copy(U)
    Vᵒ = copy(V)
#    L = 0.5Lani(A(0.3, 1.0, 1*(pi/2))...)
    M = maximum(norm.([a = f(pt(r, c), P) for c in CartesianIndices(U)]))
    if !homog
        L0 = 0.9*L2d()/maximum(abs.(L2d()))
    else
        L0 = L5()
        L1 = L5()
    end

    for iter in 0:iters
        if iter % 50 == 0
            println(iter)
            Ushow[] = U
            Vshow[] = V
            yield()
        end

        for c in CartesianIndices(U)

            x = pt(r, c)
            a = f(x, P)

            u = U[c]
            v = V[c]
            #L1 = Lani(1.0, 0.5, a[1], a[2])
            if !homog
                L1 = Lani(1.0, 0.65, a[1], a[2])
            else
                L1 = L3()
            end
            #L1 = Lani(1.0, 0.45, a[1], a[2])

            Λu = laplace(U, L0, c)
            Λv = laplace(V, L1, c)
            #L1 = Lani(.2, 1.0, a[1], a[2])
            #Λu = laplace(U, L1, c)
            #Λv = laplace(V, L0, c)
            z = u*v*v
            m = 1.0
            if scale
                m = (1 + 5norm(a)/M)/(1 + 5)
            end
            F1, K1 = F, K
            if vary
                F1, K1 = toFK(F, K, x, reparam=reparam)
            end
            if anneal
                K1 = K1*10/(10 + 500/iter) # nice at 10/(10 + 500/3000)
            end
            Δu = m*du * Λu - z + F1*(1.0 - u)
            Δv = m*dv * Λv + z - (K1+F1)*(v)

            Uᵒ[c] = u + Δu*dt
            Vᵒ[c] = v + Δv*dt
        end

        U, Uᵒ = Uᵒ, U
        V, Vᵒ = Vᵒ, V

        #
    end
end

@test laplace(U, L, (2,2)) ≈ sum(U[CartesianIndices(L)] .* L)


p = Scene(resolution=(1000,1000), limits=limits)

scene = p

on(events(scene).mousebuttons) do drag
    if ispressed(scene, Mouse.left)
        global x_ = to_world(scene, Point2f0(scene.events.mouseposition[]))
        #println(x_...)
        global FK = toFK(F, K, x_, reparam=reparam)
        println("F = ", round(FK[1], digits=3), "; K = ", round(FK[2], digits=3))

    end
end

#react!(U, V, f, P, (F, K, du, dv, dt), iters=1,)

Ushow = Node(U)
Vshow = Node(V)

heatmap!(p, r..., Vshow)
display(p)
sleep(0.1)
anneal = true
homog = false
scale = true
reparam = false

react!(U, V, f, P, (F, K, du, dv, dt),  vary=vary, anneal=anneal, scale=scale, reparam=reparam, homog=homog)
#heatmap!(p, r..., U)

display(p)

if false
    p2 = Scene(resolution=(1000,1000))
    image!(p2, r..., V)
    save("fingerprint.png", p2)
end


# react!(U, V, f, P, (0.057, 0.062, du, 0.9du, dt),  vary=vary, anneal=anneal, scale=scale, reparam=reparam, homog=homog)

#react!(U, V, f, P, (0.096, 0.0597, du, 0.85du, dt),  vary=false, anneal=false, scale=scale, reparam=reparam, iters=40, homog=homog); Vshow[] = V
