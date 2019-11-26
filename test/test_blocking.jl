function blocking_test_prep(obs=ℝ.([1.0, 1.2, 0.8, 1.3, 2.0]),
                            tt=[0.0, 1.0, 1.5, 2.3, 4.0],
                            knots=collect(1:length(obs)-2)[1:1:end],
                            change_pt_buffer=100)
    θ₀ = [10.0, -8.0, 25.0, 0.0, 3.0]
    P˟ = BSI.FitzhughDiffusion(θ₀...)
    P̃ = [BSI.FitzhughDiffusionAux(θ₀..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
         in zip(tt[1:end-1], tt[2:end], obs[1:end-1], obs[2:end])]
    L = @SMatrix [1. 0.]
    Σdiagel = 10^(-10)
    Σ = @SMatrix [Σdiagel]

    Ls = [L for _ in P̃]
    Σs = [Σ for _ in P̃]
    τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
    m = length(obs) - 1
    P = Array{BSI.ContinuousTimeProcess,1}(undef,m)
    dt = 1/50
    for i in m:-1:1
        num_pts = Int64(ceil((tt[i+1]-tt[i])/dt))+1
        t = τ(tt[i], tt[i+1]).( range(tt[i], stop=tt[i+1], length=num_pts) )
        P[i] = ( (i==m) ? BSI.GuidPropBridge(Float64, t, P˟, P̃[i], Ls[i], obs[i+1], Σs[i];
                                         change_pt=BSI.NoChangePt(change_pt_buffer),
                                         solver=BSI.Vern7()) :
                          BSI.GuidPropBridge(Float64, t, P˟, P̃[i], Ls[i], obs[i+1], Σs[i],
                                         P[i+1].H[1], P[i+1].Hν[1], P[i+1].c[1];
                                         change_pt=BSI.NoChangePt(change_pt_buffer),
                                         solver=BSI.Vern7()) )
    end
    create_blocks(BSI.ChequeredBlocking(), P,
                  (knots, 10^(-7), BSI.SimpleChangePt(change_pt_buffer)))
end

@testset "blocking object" begin
    obs = ℝ.([1.0, 1.2, 0.8, 1.3, 2.0])
    tt = [0.0, 1.0, 1.5, 2.3, 4.0]
    L = @SMatrix [1. 0.]
    ϵ = 10^(-7)
    Σ = @SMatrix [10^(-10)]

    blocks = blocking_test_prep(obs, tt)

    @testset "validity of initial set-up" begin
        @test blocks[1].knots == [1, 3]
        @test blocks[2].knots == [2]
        @test blocks[1].blocks == [[1], [2, 3], [4]]
        @test blocks[2].blocks == [[1, 2], [3, 4]]
        @test blocks[1].change_pts == [BSI.SimpleChangePt(100), BSI.NoChangePt(100), BSI.SimpleChangePt(100), BSI.NoChangePt(100)]
        @test blocks[2].change_pts == [BSI.NoChangePt(100), BSI.SimpleChangePt(100), BSI.NoChangePt(100), BSI.NoChangePt(100)]
        @test blocks[1].vs == blocks[2].vs == obs[2:end]
        @test blocks[1].Ls == [I, L, I, L]
        @test blocks[2].Ls == [L, I, L, L]
        @test blocks[1].Σs == [I*ϵ, Σ, I*ϵ, Σ]
        @test blocks[2].Σs == [Σ, I*ϵ, Σ, Σ]
        @test length(blocks[1]) == 3
        @test length(blocks[2]) == 2
        @test length(NoBlocking()) == 1
    end
end
