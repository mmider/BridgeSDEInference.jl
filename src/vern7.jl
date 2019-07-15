# NOTE: Adapted from the implementation found in Julia package
# `DifferentialEquations.jl`

using Parameters


@with_kw struct Vern7Tableau
    c₂::Float64 = 0.005
    c₃::Float64 = 0.10888888888888888
    c₄::Float64 = 0.16333333333333333
    c₅::Float64 = 0.4555
    c₆::Float64 = 0.6095094489978381
    c₇::Float64 = 0.884
    c₈::Float64 = 0.925
    a₂₁::Float64 = 0.005
    a₃₁::Float64 = -1.07679012345679
    a₃₂::Float64 = 1.185679012345679
    a₄₁::Float64 = 0.04083333333333333
    a₄₃::Float64 = 0.1225
    a₅₁::Float64 = 0.6389139236255726
    a₅₃::Float64 = -2.455672638223657
    a₅₄::Float64 = 2.272258714598084
    a₆₁::Float64 = -2.6615773750187572
    a₆₃::Float64 = 10.804513886456137
    a₆₄::Float64 = -8.3539146573962
    a₆₅::Float64 = 0.820487594956657
    a₇₁::Float64 = 6.067741434696772
    a₇₃::Float64 = -24.711273635911088
    a₇₄::Float64 = 20.427517930788895
    a₇₅::Float64 = -1.9061579788166472
    a₇₆::Float64 = 1.006172249242068
    a₈₁::Float64 = 12.054670076253203
    a₈₃::Float64 = -49.75478495046899
    a₈₄::Float64 = 41.142888638604674
    a₈₅::Float64 = -4.461760149974004
    a₈₆::Float64 = 2.042334822239175
    a₈₇::Float64 = -0.09834843665406107
    a₉₁::Float64 = 10.138146522881808
    a₉₃::Float64 = -42.6411360317175
    a₉₄::Float64 = 35.76384003992257
    a₉₅::Float64 = -4.3480228403929075
    a₉₆::Float64 = 2.0098622683770357
    a₉₇::Float64 = 0.3487490460338272
    a₉₈::Float64 = -0.27143900510483127
    a₁₀₁::Float64 = -45.030072034298676
    a₁₀₃::Float64 = 187.3272437654589
    a₁₀₄::Float64 = -154.02882369350186
    a₁₀₅::Float64 = 18.56465306347536
    a₁₀₆::Float64 = -7.141809679295079
    a₁₀₇::Float64 = 1.3088085781613787
    b₁::Float64 = 0.04715561848627222
    b₄::Float64 = 0.25750564298434153
    b₅::Float64 = 0.26216653977412624
    b₆::Float64 = 0.15216092656738558
    b₇::Float64 = 0.4939969170032485
    b₈::Float64 = -0.29430311714032503
    b₉::Float64 = 0.08131747232495111
    b̃₁::Float64 = 0.002547011879931045
    b̃₄::Float64 = -0.00965839487279575
    b̃₅::Float64 = 0.04206470975639691
    b̃₆::Float64 = -0.0666822437469301
    b̃₇::Float64 = 0.2650097464621281
    b̃₈::Float64 = -0.29430311714032503
    b̃₉::Float64 = 0.08131747232495111
    b̃₁₀::Float64 = -0.02029518466335628
end


function vern7(f, t, y, dt, P, tableau)
    (@unpack c₂,c₃,c₄,c₅,c₆,c₇,c₈,a₂₁,a₃₁,a₃₂,a₄₁,a₄₃,a₅₁,a₅₃,a₅₄,a₆₁,a₆₃,a₆₄,
             a₆₅,a₇₁,a₇₃,a₇₄,a₇₅,a₇₆,a₈₁,a₈₃,a₈₄,a₈₅,a₈₆,a₈₇,a₉₁,a₉₃,a₉₄,a₉₅,
             a₉₆,a₉₇,a₉₈,b₁,b₄,b₅,b₆,b₇,b₈,b₉ = tableau)
    k1 = f(t, y, P)
    k2 = f(t + c₂*dt, y + dt*a₂₁*k1, P)
    k3 = f(t + c₃*dt, y + dt*(a₃₁*k1 + a₃₂*k2), P)
    k4 = f(t + c₄*dt, y + dt*(a₄₁*k1 +        + a₄₃*k3), P)
    k5 = f(t + c₅*dt, y + dt*(a₅₁*k1 +        + a₅₃*k3 + a₅₄*k4), P)
    k6 = f(t + c₆*dt, y + dt*(a₆₁*k1 +        + a₆₃*k3 + a₆₄*k4 + a₆₅*k5), P)
    k7 = f(t + c₇*dt, y + dt*(a₇₁*k1 +        + a₇₃*k3 + a₇₄*k4 + a₇₅*k5
                              + a₇₆*k6), P)
    k8 = f(t + c₈*dt, y + dt*(a₈₁*k1 +        + a₈₃*k3 + a₈₄*k4 + a₈₅*k5
                              + a₈₆*k6 + a₈₇*k7), P)
    k9 = f(t + dt, y + dt*(a₉₁*k1 +        + a₉₃*k3 + a₉₄*k4 + a₉₅*k5
                           + a₉₆*k6 + a₉₇*k7 + a₉₈*k8), P)
    y + dt*(b₁*k1 + b₄*k4 + b₅*k5 + b₆*k6 + b₇*k7 + b₈*k8 + b₉*k9)
end

struct Vern7 <: ODESolverType end

#=
        FOUR components
=#
function update(::Vern7, fs, t, A, B, C, D, dt, P, tableau)
    (@unpack c₂,c₃,c₄,c₅,c₆,c₇,c₈,a₂₁,a₃₁,a₃₂,a₄₁,a₄₃,a₅₁,a₅₃,a₅₄,a₆₁,a₆₃,a₆₄,
             a₆₅,a₇₁,a₇₃,a₇₄,a₇₅,a₇₆,a₈₁,a₈₃,a₈₄,a₈₅,a₈₆,a₈₇,a₉₁,a₉₃,a₉₄,a₉₅,
             a₉₆,a₉₇,a₉₈,b₁,b₄,b₅,b₆,b₇,b₈,b₉ = tableau)
    kA1 = update(fs[1], t, A, B, C, D, P)
    kB1 = update(fs[2], t, A, B, C, D, P)
    kC1 = update(fs[3], t, A, B, C, D, P)
    kD1 = update(fs[4], t, A, B, C, D, P)

    Ai = A + dt*a₂₁*kA1
    Bi = B + dt*a₂₁*kB1
    Ci = C + dt*a₂₁*kC1
    Di = D + dt*a₂₁*kD1

    kA2 = update(fs[1], t + c₂*dt, Ai, Bi, Ci, Di, P)
    kB2 = update(fs[2], t + c₂*dt, Ai, Bi, Ci, Di, P)
    kC2 = update(fs[3], t + c₂*dt, Ai, Bi, Ci, Di, P)
    kD2 = update(fs[4], t + c₂*dt, Ai, Bi, Ci, Di, P)

    Ai = A + dt*(a₃₁*kA1 + a₃₂*kA2)
    Bi = B + dt*(a₃₁*kB1 + a₃₂*kB2)
    Ci = C + dt*(a₃₁*kC1 + a₃₂*kC2)
    Di = D + dt*(a₃₁*kD1 + a₃₂*kD2)

    kA3 = update(fs[1], t + c₃*dt, Ai, Bi, Ci, Di, P)
    kB3 = update(fs[2], t + c₃*dt, Ai, Bi, Ci, Di, P)
    kC3 = update(fs[3], t + c₃*dt, Ai, Bi, Ci, Di, P)
    kD3 = update(fs[4], t + c₃*dt, Ai, Bi, Ci, Di, P)

    Ai = A + dt*(a₄₁*kA1 +         + a₄₃*kA3)
    Bi = B + dt*(a₄₁*kB1 +         + a₄₃*kB3)
    Ci = C + dt*(a₄₁*kC1 +         + a₄₃*kC3)
    Di = D + dt*(a₄₁*kD1 +         + a₄₃*kD3)

    kA4 = update(fs[1], t + c₄*dt, Ai, Bi, Ci, Di, P)
    kB4 = update(fs[2], t + c₄*dt, Ai, Bi, Ci, Di, P)
    kC4 = update(fs[3], t + c₄*dt, Ai, Bi, Ci, Di, P)
    kD4 = update(fs[4], t + c₄*dt, Ai, Bi, Ci, Di, P)

    Ai = A + dt*(a₅₁*kA1 +         + a₅₃*kA3 + a₅₄*kA4)
    Bi = B + dt*(a₅₁*kB1 +         + a₅₃*kB3 + a₅₄*kB4)
    Ci = C + dt*(a₅₁*kC1 +         + a₅₃*kC3 + a₅₄*kC4)
    Di = D + dt*(a₅₁*kD1 +         + a₅₃*kD3 + a₅₄*kD4)

    kA5 = update(fs[1], t + c₅*dt, Ai, Bi, Ci, Di, P)
    kB5 = update(fs[2], t + c₅*dt, Ai, Bi, Ci, Di, P)
    kC5 = update(fs[3], t + c₅*dt, Ai, Bi, Ci, Di, P)
    kD5 = update(fs[4], t + c₅*dt, Ai, Bi, Ci, Di, P)

    Ai = A + dt*(a₆₁*kA1 +         + a₆₃*kA3 + a₆₄*kA4 + a₆₅*kA5)
    Bi = B + dt*(a₆₁*kB1 +         + a₆₃*kB3 + a₆₄*kB4 + a₆₅*kB5)
    Ci = C + dt*(a₆₁*kC1 +         + a₆₃*kC3 + a₆₄*kC4 + a₆₅*kC5)
    Di = D + dt*(a₆₁*kD1 +         + a₆₃*kD3 + a₆₄*kD4 + a₆₅*kD5)

    kA6 = update(fs[1], t + c₆*dt, Ai, Bi, Ci, Di, P)
    kB6 = update(fs[2], t + c₆*dt, Ai, Bi, Ci, Di, P)
    kC6 = update(fs[3], t + c₆*dt, Ai, Bi, Ci, Di, P)
    kD6 = update(fs[4], t + c₆*dt, Ai, Bi, Ci, Di, P)

    Ai = A + dt*(a₇₁*kA1 +         + a₇₃*kA3 + a₇₄*kA4 + a₇₅*kA5 + a₇₆*kA6)
    Bi = B + dt*(a₇₁*kB1 +         + a₇₃*kB3 + a₇₄*kB4 + a₇₅*kB5 + a₇₆*kB6)
    Ci = C + dt*(a₇₁*kC1 +         + a₇₃*kC3 + a₇₄*kC4 + a₇₅*kC5 + a₇₆*kC6)
    Di = D + dt*(a₇₁*kD1 +         + a₇₃*kD3 + a₇₄*kD4 + a₇₅*kD5 + a₇₆*kD6)

    kA7 = update(fs[1], t + c₇*dt, Ai, Bi, Ci, Di, P)
    kB7 = update(fs[2], t + c₇*dt, Ai, Bi, Ci, Di, P)
    kC7 = update(fs[3], t + c₇*dt, Ai, Bi, Ci, Di, P)
    kD7 = update(fs[4], t + c₇*dt, Ai, Bi, Ci, Di, P)

    Ai = A + dt*(a₈₁*kA1 +         + a₈₃*kA3 + a₈₄*kA4 + a₈₅*kA5 + a₈₆*kA6
                 + a₈₇*kA7)
    Bi = B + dt*(a₈₁*kB1 +         + a₈₃*kB3 + a₈₄*kB4 + a₈₅*kB5 + a₈₆*kB6
                 + a₈₇*kB7)
    Ci = C + dt*(a₈₁*kC1 +         + a₈₃*kC3 + a₈₄*kC4 + a₈₅*kC5 + a₈₆*kC6
                 + a₈₇*kC7)
    Di = D + dt*(a₈₁*kD1 +         + a₈₃*kD3 + a₈₄*kD4 + a₈₅*kD5 + a₈₆*kD6
                 + a₈₇*kD7)

    kA8 = update(fs[1], t + c₈*dt, Ai, Bi, Ci, Di, P)
    kB8 = update(fs[2], t + c₈*dt, Ai, Bi, Ci, Di, P)
    kC8 = update(fs[3], t + c₈*dt, Ai, Bi, Ci, Di, P)
    kD8 = update(fs[4], t + c₈*dt, Ai, Bi, Ci, Di, P)

    Ai = A + dt*(a₉₁*kA1 +         + a₉₃*kA3 + a₉₄*kA4 + a₉₅*kA5 + a₉₆*kA6
                 + a₉₇*kA7 + a₉₈*kA8)
    Bi = B + dt*(a₉₁*kB1 +         + a₉₃*kB3 + a₉₄*kB4 + a₉₅*kB5 + a₉₆*kB6
                 + a₉₇*kB7 + a₉₈*kB8)
    Ci = C + dt*(a₉₁*kC1 +         + a₉₃*kC3 + a₉₄*kC4 + a₉₅*kC5 + a₉₆*kC6
                 + a₉₇*kC7 + a₉₈*kC8)
    Di = D + dt*(a₉₁*kD1 +         + a₉₃*kD3 + a₉₄*kD4 + a₉₅*kD5 + a₉₆*kD6
                 + a₉₇*kD7 + a₉₈*kD8)

    kA9 = update(fs[1], t + dt, Ai, Bi, Ci, Di, P)
    kB9 = update(fs[2], t + dt, Ai, Bi, Ci, Di, P)
    kC9 = update(fs[3], t + dt, Ai, Bi, Ci, Di, P)
    kD9 = update(fs[4], t + dt, Ai, Bi, Ci, Di, P)

    (A + dt*(b₁*kA1 + b₄*kA4 + b₅*kA5 + b₆*kA6 + b₇*kA7 + b₈*kA8 + b₉*kA9),
     B + dt*(b₁*kB1 + b₄*kB4 + b₅*kB5 + b₆*kB6 + b₇*kB7 + b₈*kB8 + b₉*kB9),
     C + dt*(b₁*kC1 + b₄*kC4 + b₅*kC5 + b₆*kC6 + b₇*kC7 + b₈*kC8 + b₉*kC9),
     D + dt*(b₁*kD1 + b₄*kD4 + b₅*kD5 + b₆*kD6 + b₇*kD7 + b₈*kD8 + b₉*kD9))
end


#=
        THREE components
=#
function update(::Vern7, fs, t, A, B, C, dt, P, tableau)
    (@unpack c₂,c₃,c₄,c₅,c₆,c₇,c₈,a₂₁,a₃₁,a₃₂,a₄₁,a₄₃,a₅₁,a₅₃,a₅₄,a₆₁,a₆₃,a₆₄,
             a₆₅,a₇₁,a₇₃,a₇₄,a₇₅,a₇₆,a₈₁,a₈₃,a₈₄,a₈₅,a₈₆,a₈₇,a₉₁,a₉₃,a₉₄,a₉₅,
             a₉₆,a₉₇,a₉₈,b₁,b₄,b₅,b₆,b₇,b₈,b₉ = tableau)
    kA1 = update(fs[1], t, A, B, C, P)
    kB1 = update(fs[2], t, A, B, C, P)
    kC1 = update(fs[3], t, A, B, C, P)

    Ai = A + dt*a₂₁*kA1
    Bi = B + dt*a₂₁*kB1
    Ci = C + dt*a₂₁*kC1

    kA2 = update(fs[1], t + c₂*dt, Ai, Bi, Ci, P)
    kB2 = update(fs[2], t + c₂*dt, Ai, Bi, Ci, P)
    kC2 = update(fs[3], t + c₂*dt, Ai, Bi, Ci, P)

    Ai = A + dt*(a₃₁*kA1 + a₃₂*kA2)
    Bi = B + dt*(a₃₁*kB1 + a₃₂*kB2)
    Ci = C + dt*(a₃₁*kC1 + a₃₂*kC2)

    kA3 = update(fs[1], t + c₃*dt, Ai, Bi, Ci, P)
    kB3 = update(fs[2], t + c₃*dt, Ai, Bi, Ci, P)
    kC3 = update(fs[3], t + c₃*dt, Ai, Bi, Ci, P)

    Ai = A + dt*(a₄₁*kA1 +         + a₄₃*kA3)
    Bi = B + dt*(a₄₁*kB1 +         + a₄₃*kB3)
    Ci = C + dt*(a₄₁*kC1 +         + a₄₃*kC3)

    kA4 = update(fs[1], t + c₄*dt, Ai, Bi, Ci, P)
    kB4 = update(fs[2], t + c₄*dt, Ai, Bi, Ci, P)
    kC4 = update(fs[3], t + c₄*dt, Ai, Bi, Ci, P)

    Ai = A + dt*(a₅₁*kA1 +         + a₅₃*kA3 + a₅₄*kA4)
    Bi = B + dt*(a₅₁*kB1 +         + a₅₃*kB3 + a₅₄*kB4)
    Ci = C + dt*(a₅₁*kC1 +         + a₅₃*kC3 + a₅₄*kC4)

    kA5 = update(fs[1], t + c₅*dt, Ai, Bi, Ci, P)
    kB5 = update(fs[2], t + c₅*dt, Ai, Bi, Ci, P)
    kC5 = update(fs[3], t + c₅*dt, Ai, Bi, Ci, P)

    Ai = A + dt*(a₆₁*kA1 +         + a₆₃*kA3 + a₆₄*kA4 + a₆₅*kA5)
    Bi = B + dt*(a₆₁*kB1 +         + a₆₃*kB3 + a₆₄*kB4 + a₆₅*kB5)
    Ci = C + dt*(a₆₁*kC1 +         + a₆₃*kC3 + a₆₄*kC4 + a₆₅*kC5)

    kA6 = update(fs[1], t + c₆*dt, Ai, Bi, Ci, P)
    kB6 = update(fs[2], t + c₆*dt, Ai, Bi, Ci, P)
    kC6 = update(fs[3], t + c₆*dt, Ai, Bi, Ci, P)

    Ai = A + dt*(a₇₁*kA1 +         + a₇₃*kA3 + a₇₄*kA4 + a₇₅*kA5 + a₇₆*kA6)
    Bi = B + dt*(a₇₁*kB1 +         + a₇₃*kB3 + a₇₄*kB4 + a₇₅*kB5 + a₇₆*kB6)
    Ci = C + dt*(a₇₁*kC1 +         + a₇₃*kC3 + a₇₄*kC4 + a₇₅*kC5 + a₇₆*kC6)

    kA7 = update(fs[1], t + c₇*dt, Ai, Bi, Ci, P)
    kB7 = update(fs[2], t + c₇*dt, Ai, Bi, Ci, P)
    kC7 = update(fs[3], t + c₇*dt, Ai, Bi, Ci, P)

    Ai = A + dt*(a₈₁*kA1 +         + a₈₃*kA3 + a₈₄*kA4 + a₈₅*kA5 + a₈₆*kA6
                 + a₈₇*kA7)
    Bi = B + dt*(a₈₁*kB1 +         + a₈₃*kB3 + a₈₄*kB4 + a₈₅*kB5 + a₈₆*kB6
                 + a₈₇*kB7)
    Ci = C + dt*(a₈₁*kC1 +         + a₈₃*kC3 + a₈₄*kC4 + a₈₅*kC5 + a₈₆*kC6
                 + a₈₇*kC7)

    kA8 = update(fs[1], t + c₈*dt, Ai, Bi, Ci, P)
    kB8 = update(fs[2], t + c₈*dt, Ai, Bi, Ci, P)
    kC8 = update(fs[3], t + c₈*dt, Ai, Bi, Ci, P)

    Ai = A + dt*(a₉₁*kA1 +         + a₉₃*kA3 + a₉₄*kA4 + a₉₅*kA5 + a₉₆*kA6
                 + a₉₇*kA7 + a₉₈*kA8)
    Bi = B + dt*(a₉₁*kB1 +         + a₉₃*kB3 + a₉₄*kB4 + a₉₅*kB5 + a₉₆*kB6
                 + a₉₇*kB7 + a₉₈*kB8)
    Ci = C + dt*(a₉₁*kC1 +         + a₉₃*kC3 + a₉₄*kC4 + a₉₅*kC5 + a₉₆*kC6
                 + a₉₇*kC7 + a₉₈*kC8)

    kA9 = update(fs[1], t + dt, Ai, Bi, Ci, P)
    kB9 = update(fs[2], t + dt, Ai, Bi, Ci, P)
    kC9 = update(fs[3], t + dt, Ai, Bi, Ci, P)

    (A + dt*(b₁*kA1 + b₄*kA4 + b₅*kA5 + b₆*kA6 + b₇*kA7 + b₈*kA8 + b₉*kA9),
     B + dt*(b₁*kB1 + b₄*kB4 + b₅*kB5 + b₆*kB6 + b₇*kB7 + b₈*kB8 + b₉*kB9),
     C + dt*(b₁*kC1 + b₄*kC4 + b₅*kC5 + b₆*kC6 + b₇*kC7 + b₈*kC8 + b₉*kC9))
end

#=
        TWO components
=#
function update(::Vern7, fs, t, A, B, dt, P, tableau)
    (@unpack c₂,c₃,c₄,c₅,c₆,c₇,c₈,a₂₁,a₃₁,a₃₂,a₄₁,a₄₃,a₅₁,a₅₃,a₅₄,a₆₁,a₆₃,a₆₄,
             a₆₅,a₇₁,a₇₃,a₇₄,a₇₅,a₇₆,a₈₁,a₈₃,a₈₄,a₈₅,a₈₆,a₈₇,a₉₁,a₉₃,a₉₄,a₉₅,
             a₉₆,a₉₇,a₉₈,b₁,b₄,b₅,b₆,b₇,b₈,b₉ = tableau)
    kA1 = update(fs[1], t, A, B, P)
    kB1 = update(fs[2], t, A, B, P)

    Ai = A + dt*a₂₁*kA1
    Bi = B + dt*a₂₁*kB1

    kA2 = update(fs[1], t + c₂*dt, Ai, Bi, P)
    kB2 = update(fs[2], t + c₂*dt, Ai, Bi, P)

    Ai = A + dt*(a₃₁*kA1 + a₃₂*kA2)
    Bi = B + dt*(a₃₁*kB1 + a₃₂*kB2)

    kA3 = update(fs[1], t + c₃*dt, Ai, Bi, P)
    kB3 = update(fs[2], t + c₃*dt, Ai, Bi, P)

    Ai = A + dt*(a₄₁*kA1 +         + a₄₃*kA3)
    Bi = B + dt*(a₄₁*kB1 +         + a₄₃*kB3)

    kA4 = update(fs[1], t + c₄*dt, Ai, Bi, P)
    kB4 = update(fs[2], t + c₄*dt, Ai, Bi, P)

    Ai = A + dt*(a₅₁*kA1 +         + a₅₃*kA3 + a₅₄*kA4)
    Bi = B + dt*(a₅₁*kB1 +         + a₅₃*kB3 + a₅₄*kB4)

    kA5 = update(fs[1], t + c₅*dt, Ai, Bi, P)
    kB5 = update(fs[2], t + c₅*dt, Ai, Bi, P)

    Ai = A + dt*(a₆₁*kA1 +         + a₆₃*kA3 + a₆₄*kA4 + a₆₅*kA5)
    Bi = B + dt*(a₆₁*kB1 +         + a₆₃*kB3 + a₆₄*kB4 + a₆₅*kB5)

    kA6 = update(fs[1], t + c₆*dt, Ai, Bi, P)
    kB6 = update(fs[2], t + c₆*dt, Ai, Bi, P)

    Ai = A + dt*(a₇₁*kA1 +         + a₇₃*kA3 + a₇₄*kA4 + a₇₅*kA5 + a₇₆*kA6)
    Bi = B + dt*(a₇₁*kB1 +         + a₇₃*kB3 + a₇₄*kB4 + a₇₅*kB5 + a₇₆*kB6)

    kA7 = update(fs[1], t + c₇*dt, Ai, Bi, P)
    kB7 = update(fs[2], t + c₇*dt, Ai, Bi, P)

    Ai = A + dt*(a₈₁*kA1 +         + a₈₃*kA3 + a₈₄*kA4 + a₈₅*kA5 + a₈₆*kA6
                 + a₈₇*kA7)
    Bi = B + dt*(a₈₁*kB1 +         + a₈₃*kB3 + a₈₄*kB4 + a₈₅*kB5 + a₈₆*kB6
                 + a₈₇*kB7)

    kA8 = update(fs[1], t + c₈*dt, Ai, Bi, P)
    kB8 = update(fs[2], t + c₈*dt, Ai, Bi, P)

    Ai = A + dt*(a₉₁*kA1 +         + a₉₃*kA3 + a₉₄*kA4 + a₉₅*kA5 + a₉₆*kA6
                 + a₉₇*kA7 + a₉₈*kA8)
    Bi = B + dt*(a₉₁*kB1 +         + a₉₃*kB3 + a₉₄*kB4 + a₉₅*kB5 + a₉₆*kB6
                 + a₉₇*kB7 + a₉₈*kB8)

    kA9 = update(fs[1], t + dt, Ai, Bi, P)
    kB9 = update(fs[2], t + dt, Ai, Bi, P)

    (A + dt*(b₁*kA1 + b₄*kA4 + b₅*kA5 + b₆*kA6 + b₇*kA7 + b₈*kA8 + b₉*kA9),
     B + dt*(b₁*kB1 + b₄*kB4 + b₅*kB5 + b₆*kB6 + b₇*kB7 + b₈*kB8 + b₉*kB9))
end



"""
    createTableau(::Vern7)

Tableau of coefficients for the `Vern7` ODE solver
"""
createTableau(::Vern7) = Vern7Tableau()
