function ralston3(f, t, y, dt, P)
    k1 = f(t, y, P)
    k2 = f(t + 1/2*dt, y + 1/2*dt*k1, P)
    k3 = f(t + 3/4*dt, y + 3/4*dt*k2, P)
    y + dt*(2/9*k1 + 1/3*k2 + 4/9*k3)
end


struct Ralston3 <: ODESolverType end

function update(::Ralston3, fs, t, A, B, C, D, dt, P, tableau=NaN)
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

    Ai = A + 3/4*dt*kA2
    Bi = B + 3/4*dt*kB2
    Ci = C + 3/4*dt*kC2
    Di = D + 3/4*dt*kD2

    kA3 = update(fs[1], t + 3/4*dt, Ai, Bi, Ci, Di, P)
    kB3 = update(fs[2], t + 3/4*dt, Ai, Bi, Ci, Di, P)
    kC3 = update(fs[3], t + 3/4*dt, Ai, Bi, Ci, Di, P)
    kD3 = update(fs[4], t + 3/4*dt, Ai, Bi, Ci, Di, P)

    (A + dt*(2/9*kA1 + 1/3*kA2 + 4/9*kA3), B + dt*(2/9*kB1 + 1/3*kB2 + 4/9*kB3),
     C + dt*(2/9*kC1 + 1/3*kC2 + 4/9*kC3), D + dt*(2/9*kD1 + 1/3*kD2 + 4/9*kD3))
end
