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

struct Tsit5 <: ODESolverType end

function tsit5(::Tsit5, f::Function, t, y, dt, P, tableau)
    (@unpack c₁,c₂,c₃,c₄,c₅,c₆,a₂₁,a₃₁,a₃₂,a₄₁,a₄₂,a₄₃,a₅₁,a₅₂,a₅₃,a₅₄,a₆₁,a₆₂,
             a₆₃,a₆₄,a₆₅,a₇₁,a₇₂,a₇₃,a₇₄,a₇₅,a₇₆ = tableau)
    k1 = f(t, y, P)
    k2 = f(t + c₁*dt, y .+ (dt*a₂₁).*k1, P)
    k3 = f(t + c₂*dt, y .+ (dt*a₃₁).*k1 .+ (dt*a₃₂).*k2, P)
    k4 = f(t + c₃*dt, y .+ (dt*a₄₁).*k1 .+ (dt*a₄₂).*k2 .+ (dt*a₄₃).*k3, P)
    k5 = f(t + c₄*dt, y .+ (dt*a₅₁).*k1 .+ (dt*a₅₂).*k2 .+ (dt*a₅₃).*k3
                        .+ (dt*a₅₄).*k4, P)
    k6 = f(t + c₅*dt, y .+ (dt*a₆₁).*k1 .+ (dt*a₆₂).*k2 .+ (dt*a₆₃).*k3
                        .+ (dt*a₆₄).*k4 .+ (dt*a₆₅).*k5, P)
    (y .+ (dt*a₇₁).*k1 .+ (dt*a₇₂).*k2 .+ (dt*a₇₃).*k3 .+ (dt*a₇₄).*k4
       .+ (dt*a₇₅).*k5 .+ (dt*a₇₆).*k6)
end
