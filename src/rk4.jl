
function rk4(f, t, y, dt, P, k)
    k1 = f(t, y, P)
    k2 = f(t + 1/2*dt, y + 1/2*dt*k1, P)
    k3 = f(t + 1/2*dt, y + 1/2*dt*k2, P)
    k4 = f(t + dt, y + dt*k3, P)
    y + dt*(k1+2*k2+2*k3+k4)/6
end

struct Rk4 <: ODESolverType end

function update(::Rk4, fs, t, A, B, C, D, dt, P, tableau=NaN)
    kA1 = update(fs[1], t, A, B, C, D, P)
    kB1 = update(fs[2], t, A, B, C, D, P)
    kC1 = update(fs[3], t, A, B, C, D, P)
    kD1 = update(fs[4], t, A, B, C, D, P)

    Ai = A + 1/2*dt*kA1
    Bi = B + 1/2*dt*kB1
    Ci = C + 1/2*dt*kC1
    Di = D + 1/2*dt*kD1

    kA2 = update(fs[1], t + 1/2*dt, Ai, Bi, Ci, Di, P)
    kB2 = update(fs[2], t + 1/2*dt, Ai, Bi, Ci, Di, P)
    kC2 = update(fs[3], t + 1/2*dt, Ai, Bi, Ci, Di, P)
    kD2 = update(fs[4], t + 1/2*dt, Ai, Bi, Ci, Di, P)

    Ai = A + 1/2*dt*kA2
    Bi = B + 1/2*dt*kB2
    Ci = C + 1/2*dt*kC2
    Di = D + 1/2*dt*kD2

    kA3 = update(fs[1], t + 1/2*dt, Ai, Bi, Ci, Di, P)
    kB3 = update(fs[2], t + 1/2*dt, Ai, Bi, Ci, Di, P)
    kC3 = update(fs[3], t + 1/2*dt, Ai, Bi, Ci, Di, P)
    kD3 = update(fs[4], t + 1/2*dt, Ai, Bi, Ci, Di, P)

    Ai = A + dt*kA3
    Bi = B + dt*kB3
    Ci = C + dt*kC3
    Di = D + dt*kD3

    kA4 = update(fs[1], t + dt, Ai, Bi, Ci, Di, P)
    kB4 = update(fs[2], t + dt, Ai, Bi, Ci, Di, P)
    kC4 = update(fs[3], t + dt, Ai, Bi, Ci, Di, P)
    kD4 = update(fs[4], t + dt, Ai, Bi, Ci, Di, P)

    (A + dt*(kA1 + 2*kA2 + 2*kA3 + kA4)/6, B + dt*(kB1 + 2*kB2 + 2*kB3 + kB4)/6,
     C + dt*(kC1 + 2*kC2 + 2*kC3 + kC4)/6, D + dt*(kD1 + 2*kD2 + 2*kD3 + kD4)/6)
end
