# NOTE: Adapted from the implementation found in Julia package
# `DifferentialEquations.jl`


using Parameters

@with_kw struct Tsit5Tableau
    c₁::Float64 = 0.161
    c₂::Float64 = 0.327
    c₃::Float64 = 0.9
    c₄::Float64 = 0.9800255409045097
    c₅::Float64 = 1
    c₆::Float64 = 1
    a₂₁::Float64 = 0.161
    a₃₁::Float64 = -0.008480655492356989
    a₃₂::Float64 = 0.335480655492357
    a₄₁::Float64 = 2.8971530571054935
    a₄₂::Float64 = -6.359448489975075
    a₄₃::Float64 = 4.3622954328695815
    a₅₁::Float64 = 5.325864828439257
    a₅₂::Float64 = -11.748883564062828
    a₅₃::Float64 = 7.4955393428898365
    a₅₄::Float64 = -0.09249506636175525
    a₆₁::Float64 = 5.86145544294642
    a₆₂::Float64 = -12.92096931784711
    a₆₃::Float64 = 8.159367898576159
    a₆₄::Float64 = -0.071584973281401
    a₆₅::Float64 = -0.028269050394068383
    a₇₁::Float64 = 0.09646076681806523
    a₇₂::Float64 = 0.01
    a₇₃::Float64 = 0.4798896504144996
    a₇₄::Float64 = 1.379008574103742
    a₇₅::Float64 = -3.290069515436081
    a₇₆::Float64 = 2.324710524099774
    b̃₁::Float64 = -0.00178001105222577714
    b̃₂::Float64 = -0.0008164344596567469
    b̃₃::Float64 = 0.007880878010261995
    b̃₄::Float64 = -0.1447110071732629
    b̃₅::Float64 = 0.5823571654525552
    b̃₆::Float64 = -0.45808210592918697
    b̃₇::Float64 = 0.015151515151515152
end

function tsit5(f, t, y, dt, P, tableau)
    (@unpack c₁,c₂,c₃,c₄,c₅,c₆,a₂₁,a₃₁,a₃₂,a₄₁,a₄₂,a₄₃,a₅₁,a₅₂,a₅₃,a₅₄,a₆₁,a₆₂,
             a₆₃,a₆₄,a₆₅,a₇₁,a₇₂,a₇₃,a₇₄,a₇₅,a₇₆ = tableau)
    k1 = f(t, y, P)
    k2 = f(t + c₁*dt, y + dt*a₂₁*k1, P)
    k3 = f(t + c₂*dt, y + dt*(a₃₁*k1 + a₃₂*k2), P)
    k4 = f(t + c₃*dt, y + dt*(a₄₁*k1 + a₄₂*k2 + a₄₃*k3), P)
    k5 = f(t + c₄*dt, y + dt*(a₅₁*k1 + a₅₂*k2 + a₅₃*k3 + a₅₄*k4), P)
    k6 = f(t + c₅*dt, y + dt*(a₆₁*k1 + a₆₂*k2 + a₆₃*k3 + a₆₄*k4 + a₆₅*k5), P)
    y + dt*(a₇₁*k1 + a₇₂*k2 + a₇₃*k3 + a₇₄*k4 + a₇₅*k5 + a₇₆*k6)
end

struct Tsit5 <: ODESolverType end

function update(::Tsit5, fs, t, A, B, C, D, dt, P, tableau)
    (@unpack c₁,c₂,c₃,c₄,c₅,c₆,a₂₁,a₃₁,a₃₂,a₄₁,a₄₂,a₄₃,a₅₁,a₅₂,a₅₃,a₅₄,a₆₁,a₆₂,
             a₆₃,a₆₄,a₆₅,a₇₁,a₇₂,a₇₃,a₇₄,a₇₅,a₇₆ = tableau)
    kA1 = update(fs[1], t, A, B, C, D, P)
    kB1 = update(fs[2], t, A, B, C, D, P)
    kC1 = update(fs[3], t, A, B, C, D, P)
    kD1 = update(fs[4], t, A, B, C, D, P)

    Ai = A + dt*a₂₁*kA1
    Bi = B + dt*a₂₁*kB1
    Ci = C + dt*a₂₁*kC1
    Di = D + dt*a₂₁*kD1

    kA2 = update(fs[1], t + c₁*dt, Ai, Bi, Ci, Di, P)
    kB2 = update(fs[2], t + c₁*dt, Ai, Bi, Ci, Di, P)
    kC2 = update(fs[3], t + c₁*dt, Ai, Bi, Ci, Di, P)
    kD2 = update(fs[4], t + c₁*dt, Ai, Bi, Ci, Di, P)

    Ai = A + dt*(a₃₁*kA1 + a₃₂*kA2)
    Bi = B + dt*(a₃₁*kB1 + a₃₂*kB2)
    Ci = C + dt*(a₃₁*kC1 + a₃₂*kC2)
    Di = D + dt*(a₃₁*kD1 + a₃₂*kD2)

    kA3 = update(fs[1], t + c₂*dt, Ai, Bi, Ci, Di, P)
    kB3 = update(fs[2], t + c₂*dt, Ai, Bi, Ci, Di, P)
    kC3 = update(fs[3], t + c₂*dt, Ai, Bi, Ci, Di, P)
    kD3 = update(fs[4], t + c₂*dt, Ai, Bi, Ci, Di, P)

    Ai = A + dt*(a₄₁*kA1 + a₄₂*kA2 + a₄₃*kA3)
    Bi = B + dt*(a₄₁*kB1 + a₄₂*kB2 + a₄₃*kB3)
    Ci = C + dt*(a₄₁*kC1 + a₄₂*kC2 + a₄₃*kC3)
    Di = D + dt*(a₄₁*kD1 + a₄₂*kD2 + a₄₃*kD3)

    kA4 = update(fs[1], t + c₃*dt, Ai, Bi, Ci, Di, P)
    kB4 = update(fs[2], t + c₃*dt, Ai, Bi, Ci, Di, P)
    kC4 = update(fs[3], t + c₃*dt, Ai, Bi, Ci, Di, P)
    kD4 = update(fs[4], t + c₃*dt, Ai, Bi, Ci, Di, P)

    Ai = A + dt*(a₅₁*kA1 + a₅₂*kA2 + a₅₃*kA3 + a₅₄*kA4)
    Bi = B + dt*(a₅₁*kB1 + a₅₂*kB2 + a₅₃*kB3 + a₅₄*kB4)
    Ci = C + dt*(a₅₁*kC1 + a₅₂*kC2 + a₅₃*kC3 + a₅₄*kC4)
    Di = D + dt*(a₅₁*kD1 + a₅₂*kD2 + a₅₃*kD3 + a₅₄*kD4)

    kA5 = update(fs[1], t + c₄*dt, Ai, Bi, Ci, Di, P)
    kB5 = update(fs[2], t + c₄*dt, Ai, Bi, Ci, Di, P)
    kC5 = update(fs[3], t + c₄*dt, Ai, Bi, Ci, Di, P)
    kD5 = update(fs[4], t + c₄*dt, Ai, Bi, Ci, Di, P)

    Ai = A + dt*(a₆₁*kA1 + a₆₂*kA2 + a₆₃*kA3 + a₆₄*kA4 + a₆₅*kA5)
    Bi = B + dt*(a₆₁*kB1 + a₆₂*kB2 + a₆₃*kB3 + a₆₄*kB4 + a₆₅*kB5)
    Ci = C + dt*(a₆₁*kC1 + a₆₂*kC2 + a₆₃*kC3 + a₆₄*kC4 + a₆₅*kC5)
    Di = D + dt*(a₆₁*kD1 + a₆₂*kD2 + a₆₃*kD3 + a₆₄*kD4 + a₆₅*kD5)

    kA6 = update(fs[1], t + c₅*dt, Ai, Bi, Ci, Di, P)
    kB6 = update(fs[2], t + c₅*dt, Ai, Bi, Ci, Di, P)
    kC6 = update(fs[3], t + c₅*dt, Ai, Bi, Ci, Di, P)
    kD6 = update(fs[4], t + c₅*dt, Ai, Bi, Ci, Di, P)

    (A + dt*(a₇₁*kA1 + a₇₂*kA2 + a₇₃*kA3 + a₇₄*kA4 + a₇₅*kA5 + a₇₆*kA6),
     B + dt*(a₇₁*kB1 + a₇₂*kB2 + a₇₃*kB3 + a₇₄*kB4 + a₇₅*kB5 + a₇₆*kB6),
     C + dt*(a₇₁*kC1 + a₇₂*kC2 + a₇₃*kC3 + a₇₄*kC4 + a₇₅*kC5 + a₇₆*kC6),
     D + dt*(a₇₁*kD1 + a₇₂*kD2 + a₇₃*kD3 + a₇₄*kD4 + a₇₅*kD5 + a₇₆*kD6))
end
