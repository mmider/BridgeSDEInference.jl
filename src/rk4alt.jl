
struct Rk4 <: ODESolverType end

function rk4(::Rk4, f::Function, t, y, dt, P, ::Any)
    k1 = f(t, y, P)
    k2 = f(t + 1/2*dt, y .+ (1/2*dt).*k1, P)
    k3 = f(t + 1/2*dt, y .+ (1/2*dt).*k2, P)
    k4 = f(t + dt, y .+ dt.*k3, P)
    y .+ (dt/6).*k1 .+ (dt/3).*k2 .+ (dt/3).*k3 .+ (dt/6).*k4
end
