@testset "diffusion setup object" begin
    param = :complexConjug
    θ₀ = [10.0, -5.0, 5.0, 0.0, 3.0]
    P˟ = BSI.FitzhughDiffusion(param, θ₀...)
    obs = [[0.0, 0.0], [-0.3, -0.2], [0.4, 0.2]]
    tt = [0.0, 1.0, 1.5]
    P̃ = [BSI.FitzhughDiffusionAux(param, θ₀..., tt[1], obs[1], tt[2], obs[2]),
         BSI.FitzhughDiffusionAux(param, θ₀..., tt[2], obs[2], tt[3], obs[3])]
    setup = BSI.DiffusionSetup(P˟, P̃, PartObs())

    @testset "initialisation" begin
        @test setup.P˟ == P˟
        @test setup.P̃ == P̃
        @test typeof(setup.adaptive_prop) <: Adaptation{Val{false}}
        @test setup.skip_for_save == 1
        @test !any(values(setup.setup_completion))
        @test @suppress !BSI.check_if_complete(setup, [:obs])
        @test @suppress !BSI.check_if_complete(setup, [:imput])
        @test @suppress !BSI.check_if_complete(setup, [:prior])
    end

    L = [1. 0.; 0. 1.]
    Σ = [0.5 0.0; 0.0 1.0]
    BSI.set_observations!(setup, [L, L], [Σ, 2*Σ], obs, tt)

    @testset "setting observations" begin
        @test setup.Ls == [L, L]
        @test setup.Σs == [Σ, 2*Σ]
        @test setup.obs == obs
        @test setup.obs_times == tt
        @test setup.fpt == [nothing, nothing]
        @test @suppress !BSI.check_if_complete(setup, [:imput])
        @test @suppress !BSI.check_if_complete(setup, [:prior])
        @test @suppress BSI.check_if_complete(setup, [:obs])
    end

    dt = 0.01
    τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
    BSI.set_imputation_grid!(setup, dt)

    @testset "setting imputation grid" begin
        @test setup.dt == dt
        @test setup.τ(tt[1], tt[2])(0.5*(tt[1]+tt[2])) == τ(tt[1], tt[2])(0.5*(tt[1]+tt[2]))
        @test @suppress !BSI.check_if_complete(setup, [:prior])
        @test @suppress BSI.check_if_complete(setup, [:obs, :imput])
    end

    x0_prior = BSI.KnownStartingPt(obs[1])
    BSI.set_x0_prior!(setup, x0_prior)
    @testset "setting priors" begin
        @test setup.x0_prior == x0_prior
        @test @suppress BSI.check_if_complete(setup, [:obs, :imput, :prior])
    end

    @testset "determining data type" begin
        @test BSI.determine_data_type(setup) == (SArray{Tuple{2},Float64,1,2},
                                             Float64)
    end

    BSI.prepare_containers!(setup)
    @testset "setting internal containers" begin
        @test setup.Wnr == Wiener{Float64}()
        @test eltype(setup.WW) == SamplePath{Float64}
        @test length(setup.WW) == 2
        @test eltype(setup.XX) == SamplePath{SArray{Tuple{2},Float64,1,2}}
        @test length(setup.XX) == 2
        @test setup.Ls == [SMatrix{2,2}(L), SMatrix{2,2}(L)]
        @test setup.Σs == [SMatrix{2,2}(Σ), SMatrix{2,2}(2*Σ)]
        @test setup.obs == map(x->SVector{2}(x), obs)
    end

    BSI.initialise!(Float64, setup, BSI.Vern7(), false, BSI.NoChangePt())
    @testset "initialisation of proposal law" begin
        @test length(setup.P) == 2
        @test typeof(setup.P[1].Pt) == typeof(setup.P̃[1])
        @test setup.P[1].Target == setup.P˟
    end
end
