POSSIBLE_PARAMS = [:regular, :simpleAlter, :complexAlter, :simpleConjug,
                   :complexConjug]
SRC_DIR = joinpath(Base.source_dir(), "..", "src")

parametrisation = POSSIBLE_PARAMS[5]

function blocking_test_prep(obs=ℝ.([1.0, 1.2, 0.8, 1.3, 2.0]),
                            tt=[0.0, 1.0, 1.5, 2.3, 4.0],
                            knots=collect(1:length(obs)-2)[1:1:end],
                            changePtBuffer=100)
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
        numPts = Int64(ceil((tt[i+1]-tt[i])/dt))+1
        t = τ(tt[i], tt[i+1]).( range(tt[i], stop=tt[i+1], length=numPts) )
        P[i] = ( (i==m) ? BSI.GuidPropBridge(Float64, t, P˟, P̃[i], Ls[i], obs[i+1], Σs[i];
                                         changePt=BSI.NoChangePt(changePtBuffer),
                                         solver=BSI.Vern7()) :
                          BSI.GuidPropBridge(Float64, t, P˟, P̃[i], Ls[i], obs[i+1], Σs[i],
                                         P[i+1].H[1], P[i+1].Hν[1], P[i+1].c[1];
                                         changePt=BSI.NoChangePt(changePtBuffer),
                                         solver=BSI.Vern7()) )
    end

    T = SArray{Tuple{2},Float64,1,2}
    TW = typeof(sample([0], BSI.Wiener{Float64}()))
    TX = typeof(BSI.SamplePath([], zeros(T, 0)))
    XX = Vector{TX}(undef,m)
    WW = Vector{TW}(undef,m)
    for i in 1:m
        XX[i] = BSI.SamplePath(P[i].tt, zeros(T, length(P[i].tt)))
        XX[i].yy .= [T(obs[i+1][1], i) for _ in 1:length(XX[i].yy)]
    end

    blockingParams = (knots, 10^(-7), BSI.SimpleChangePt(changePtBuffer))
    𝔅 = BSI.ChequeredBlocking(blockingParams..., P, WW, XX)
    for i in 1:m
        𝔅.XXᵒ[i].yy .= [T(obs[i+1][1], 10+i) for _ in 1:length(XX[i].yy)]
    end
    𝔅
end

@testset "blocking object" begin
    obs = ℝ.([1.0, 1.2, 0.8, 1.3, 2.0])
    tt = [0.0, 1.0, 1.5, 2.3, 4.0]
    L = @SMatrix [1. 0.]
    ϵ = 10^(-7)
    Σ = @SMatrix [10^(-10)]

    𝔅 = blocking_test_prep(obs, tt)

    @testset "validity of initial set-up" begin
        @test 𝔅.idx == 1
        @test 𝔅.knots[1] == [1, 3]
        @test 𝔅.knots[2] == [2]
        @test 𝔅.blocks[1] == [[1], [2, 3], [4]]
        @test 𝔅.blocks[2] == [[1, 2], [3, 4]]
        @test 𝔅.changePts[1] == [BSI.SimpleChangePt(100), BSI.NoChangePt(100), BSI.SimpleChangePt(100), BSI.NoChangePt(100)]
        @test 𝔅.changePts[2] == [BSI.NoChangePt(100), BSI.SimpleChangePt(100), BSI.NoChangePt(100), BSI.NoChangePt(100)]
        @test 𝔅.vs == obs[2:end]
        @test 𝔅.Ls[1] == [I, L, I, L]
        @test 𝔅.Ls[2] == [L, I, L, L]
        @test 𝔅.Σs[1] == [I*ϵ, Σ, I*ϵ, Σ]
        @test 𝔅.Σs[2] == [Σ, I*ϵ, Σ, Σ]
    end

    θ = [10.0, -8.0, 15.0, 0.0, 3.0]
    𝔅 = BSI.next(𝔅, 𝔅.XX, θ)

    @testset "validity of blocking state after calling next" begin
        @test 𝔅.idx == 2
        @testset "checking if θ has been propagated everywhere" for i in 1:length(tt)-1
            @test BSI.params(𝔅.P[i].Target) == θ
            @test BSI.params(𝔅.P[i].Pt) == θ
        end
        @test [𝔅.P[i].Σ for i in 1:length(tt)-1 ] == 𝔅.Σs[2] == [Σ, I*ϵ, Σ, Σ]
        @test [𝔅.P[i].L for i in 1:length(tt)-1 ] == 𝔅.Ls[2] == [L, I, L, L]
        @test [𝔅.P[i].v for i in 1:length(tt)-1 ] == [obs[2], 𝔅.XX[2].yy[end], obs[4], obs[5]]
        @test [𝔅.P[i].changePt for i in 1:length(tt)-1 ] == 𝔅.changePts[2] == [BSI.NoChangePt(100), BSI.SimpleChangePt(100), BSI.NoChangePt(100), BSI.NoChangePt(100)]
    end

    θᵒ = [1.0, -7.0, 10.0, 2.0, 1.0]

    @testset "checking container swaps" begin
        @testset "checking before the swap" for i in 1:length(tt)-1
            @test 𝔅.XX[i].yy[10][2] == i
            @test 𝔅.XXᵒ[i].yy[10][2] == 10 + i
        end
        for i in 1:length(tt)-1
            𝔅.XX[i], 𝔅.XXᵒ[i] = 𝔅.XXᵒ[i], 𝔅.XX[i]
        end
        @testset "checking if containers swapped" for i in 1:length(tt)-1
            @test 𝔅.XX[i].yy[10][2] == 10 + i
            @test 𝔅.XXᵒ[i].yy[10][2] == i
        end
    end

    𝔅 = BSI.next(𝔅, 𝔅.XX, θᵒ)

    @testset "validity of blocking state after second call to next" begin
        @test 𝔅.idx == 1
        @testset "checking if θᵒ has been propagated everywhere" for i in 1:length(tt)-1
            @test BSI.params(𝔅.P[i].Target) == θᵒ
            @test BSI.params(𝔅.P[i].Pt) == θᵒ
        end
        @test [𝔅.P[i].Σ for i in 1:length(tt)-1 ] == 𝔅.Σs[1] == [I*ϵ, Σ, I*ϵ, Σ]
        @test [𝔅.P[i].L for i in 1:length(tt)-1 ] == 𝔅.Ls[1] == [I, L, I, L]
        @test [𝔅.P[i].v for i in 1:length(tt)-1 ] == [𝔅.XX[1].yy[end], obs[3], 𝔅.XX[3].yy[end], obs[5]]
        @test [𝔅.P[i].changePt for i in 1:length(tt)-1 ] == 𝔅.changePts[1] == [BSI.SimpleChangePt(100), BSI.NoChangePt(100), BSI.SimpleChangePt(100), BSI.NoChangePt(100)]
    end
end
