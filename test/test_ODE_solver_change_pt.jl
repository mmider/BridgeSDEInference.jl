function change_point_test_prep(N=10000, λ=N/10)
    L = @SMatrix [1. 0.;
                  0. 1.]
    Σ = @SMatrix [10^(-5) 0.;
         0. 10^(-5)]

    θ₀ = [10.0, -8.0, 15.0, 0.0, 3.0]

    # Target law
    P˟ = FitzhughDiffusion(θ₀...)

    # Auxiliary law
    t₀ = 1.0
    T = 2.0
    x0 = ℝ{2}(-0.5, 2.25)
    xT = ℝ{2}(1.0, 0.0)
    P̃ = FitzhughDiffusionAux(θ₀..., t₀, L*x0, T, L*xT)

    τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
    dt = (T-t₀)/N
    tt = τ(t₀,T).(t₀:dt:T)

    P₁ = GuidPropBridge(eltype(x0), tt, P˟, P̃, L, L*x0, Σ;
                        changePt=NoChangePt(), solver=Vern7())

    P₂ = GuidPropBridge(eltype(x0), tt, P˟, P̃, L, L*x0, Σ;
                        changePt=SimpleChangePt(λ), solver=Vern7())
    P₁, P₂
end

@testset "change point between ODE solvers" begin

    parametrisation = POSSIBLE_PARAMS[5]
    include(joinpath(SRC_DIR, "fitzHughNagumo.jl"))
    include(joinpath(SRC_DIR, "types.jl"))
    include(joinpath(SRC_DIR, "vern7.jl"))
    include(joinpath(SRC_DIR, "guid_prop_bridge.jl"))
    N = 10000
    P₁, P₂ = change_point_test_prep(N)
    @testset "comparing H[$i]" for i in 1:div(N,20):N
        @test P₁.H[i] ≈ P₂.H[i]
    end
    @testset "comparing Hν[$i]" for i in 1:div(N,20):N
        @test P₁.Hν[i] ≈ P₂.Hν[i]
    end
    @testset "comparing c[$i]" for i in 1:div(N,20):N
        @test P₁.c[i] ≈ P₂.c[i]
    end
end
