@testset "uniform random walk" begin
    @testset "initialisation" begin
        @test_throws MethodError BSI.UniformRandomWalk([1.0])
        @test BSI.UniformRandomWalk(1.0).ϵ == 1.0
        @test BSI.UniformRandomWalk(1.0).pos == false
        @test BSI.UniformRandomWalk(2.0, true).ϵ == 2.0
        @test BSI.UniformRandomWalk(1.0, true).pos
    end
    @testset "random draw" begin
        using Random
        ϵ = 1.0
        Random.seed!(4)
        rand_value = rand(Uniform(-ϵ, ϵ))

        Random.seed!(4)
        rw = BSI.UniformRandomWalk(ϵ)
        θ = [2.0, 3.0]
        @test rand(rw, θ, Val{(false,true)}()) == θ .+ [0.0, rand_value]
        @test θ == [2.0, 3.0]

        Random.seed!(4)
        rand!(rw, θ, Val{(false,true)}())
        @test θ == [2.0, 3.0 + rand_value]
    end
    @testset "logpdf" begin
        ϵ = 1.0
        rw = BSI.UniformRandomWalk(ϵ)
        θ, θᵒ = [2.0, 3.0], [1.0, 5.0]
        @test logpdf(rw, Val{(true, false)}(), θ, θᵒ) == 0.0
        rw = BSI.UniformRandomWalk(1.0, true)
        @test logpdf(rw, Val{(true, false)}(), θ, θᵒ) == -log(2.0*ϵ)-log(θᵒ[1])
        @test_throws AssertionError logpdf(rw, Val{(true, true)}(), θ, θᵒ)
    end
    @testset "readjustment" begin
        ϵ = 1.0
        rw = BSI.UniformRandomWalk(ϵ, true)
        accpt_track = BSI.AccptTracker(0)
        for i in 1:10 register_accpt!(accpt_track, true) end
        for i in 1:10 register_accpt!(accpt_track, false) end
        δ = 0.1
        param = BSI.named_readjust((100, δ, 0.0, 999.9, 0.234, 50))
        @suppress readjust!(rw, accpt_track, param, nothing, 10, nothing)
        @test rw.ϵ == 1.0+δ

        reset!(accpt_track)
        for i in 1:2 register_accpt!(accpt_track, true) end
        for i in 1:10 register_accpt!(accpt_track, false) end
        @suppress readjust!(rw, accpt_track, param, nothing, 11, nothing)
        @test rw.ϵ == 1.0

    end
end
