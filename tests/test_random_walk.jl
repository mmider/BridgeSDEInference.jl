SRC_DIR = joinpath(Base.source_dir(), "..", "src")
include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "transition_kernels", "random_walk.jl"))

function test_init(rw, ϵ, pos)
    @test rw.ϵ == ϵ
    @test rw.pos == pos
    @test length(rw) == length(ϵ)
    @test eltype(rw) == eltype(ϵ)
end

@testset "random walk" begin
    @testset "initialisation" begin
        @test_throws AssertionError RandomWalk((1.0, 2.0), (false,))
        test_init(RandomWalk(1.0, false), (1.0,), (false,))
        test_init(RandomWalk([1.0], false), (1.0,), (false,))
        test_init(RandomWalk(1.0, (false,)), (1.0,), (false,))
        test_init(RandomWalk(1.0, 1), (1.0,), (true,))
        test_init(RandomWalk(1.0), (1.0,), (false,))
        test_init(RandomWalk(1.0, []), (1.0,), (false,))

        test_init(RandomWalk((1.0, 2.0)), (1.0,2.0), (false,false))
        test_init(RandomWalk([1.0, 2.0], []), (1.0,2.0), (false,false))
        test_init(RandomWalk([1.0, 2.0], 2), (1.0,2.0), (false,true))
        test_init(RandomWalk([1.0, 2.0], (1,2)), (1.0,2.0), (true,true))
    end
end
