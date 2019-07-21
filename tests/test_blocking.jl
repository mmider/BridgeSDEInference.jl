parametrisation = POSSIBLE_PARAMS[5]
include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "fitzHughNagumo.jl"))
include(joinpath(SRC_DIR, "guid_prop_bridge.jl"))
include(joinpath(SRC_DIR, "blocking_schedule.jl"))
include(joinpath(SRC_DIR, "vern7.jl"))

changePtBuffer=100

obs = [1.0, 1.2, 0.8, 1.2, 2.0]
tt = [0.0, 1.0, 1.5, 2.3, 4.0]
θ₀ = [10.0, -8.0, 25.0, 0.0, 3.0]
P˟ = FitzhughDiffusion(θ₀...)
P̃ = [FitzhughDiffusionAux(θ₀..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
     in zip(tt[1:end-1], tt[2:end], obs[1:end-1], obs[2:end])]
L = @SMatrix [1. 0.]
Σdiagel = 10^(-10)
Σ = @SMatrix [Σdiagel]

Ls = [L for _ in P̃]
Σs = [Σ for _ in P̃]
τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
m = length(obs) - 1
P = Array{ContinuousTimeProcess,1}(undef,m)
dt = 1/50
for i in m:-1:1
    numPts = Int64(ceil((tt[i+1]-tt[i])/dt))+1
    t = τ(tt[i], tt[i+1]).( range(tt[i], stop=tt[i+1], length=numPts) )
    P[i] = ( (i==m) ? GuidPropBridge(Float64, t, P˟, P̃[i], Ls[i], obs[i+1], Σs[i];
                                     changePt=NoChangePt(changePtBuffer),
                                     solver=Vern7()) :
                      GuidPropBridge(Float64, t, P˟, P̃[i], Ls[i], obs[i+1], Σs[i],
                                     P[i+1].H[1], P[i+1].Hν[1], P[i+1].c[1];
                                     changePt=NoChangePt(changePtBuffer),
                                     solver=Vern7()) )
end

𝔅 = ChequeredBlocking(blockingParams..., P, WW, XX)
