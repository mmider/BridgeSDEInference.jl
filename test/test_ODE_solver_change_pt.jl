

POSSIBLE_PARAMS = [:regular, :simpleAlter, :complexAlter, :simpleConjug,
                   :complexConjug]

function change_point_test_prep(N=10000, λ=N/10)
    L = @SMatrix [1. 0.;
                  0. 1.]
    Σ = @SMatrix [10^(-5) 0.;
         0. 10^(-5)]

    θ₀ = [10.0, -8.0, 15.0, 0.0, 3.0]

    param = :complexConjug
    # Target law
    P˟ = BSI.FitzhughDiffusion(param, θ₀...)

    # Auxiliary law
    t₀ = 1.0
    T = 2.0
    x0 = ℝ{2}(-0.5, 2.25)
    xT = ℝ{2}(1.0, 0.0)
    P̃ = BSI.FitzhughDiffusionAux(param, θ₀..., t₀, L*x0, T, L*xT)

    τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
    dt = (T-t₀)/N
    tt = τ(t₀,T).(t₀:dt:T)

    P₁ = BSI.GuidPropBridge(eltype(x0), tt, P˟, P̃, L, L*x0, Σ;
                        changePt=BSI.NoChangePt(), solver=BSI.Vern7())

    P₂ = BSI.GuidPropBridge(eltype(x0), tt, P˟, P̃, L, L*x0, Σ;
                        changePt=BSI.SimpleChangePt(λ), solver=BSI.Vern7())
    P₁, P₂
end

@testset "change point between ODE solvers" begin

    parametrisation = POSSIBLE_PARAMS[5]
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
