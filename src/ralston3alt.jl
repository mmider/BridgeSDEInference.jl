struct Ralston3 <: ODESolverType end

function ralston3(::Ralston3, f::Function, t, y, dt, P, ::Any)
    k1 = f(t, y, P)
    k2 = f(t + 1/2*dt, y .+ (1/2*dt).*k1, P)
    k3 = f(t + 3/4*dt, y .+ (3/4*dt).*k2, P)
    y .+ (dt*2/9).*k1 .+ (dt/3).*k2 .+ (dt*4/9).*k3
end
