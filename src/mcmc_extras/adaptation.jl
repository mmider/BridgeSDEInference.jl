import Base.resize!

struct Adaptation{TV,T}
    X::Vector{Vector{Vector{T}}}
    ρs::Vector{Float64}
    λs::Vector{Float64}
    sizes::Vector{Int64}
    skip::Int64
    N::Vector{Int64}

    function Adaptation(::T, ρs, λs, sizes_of_path_coll, skip=1) where T
        TV = Val{true}
        M = maximum(sizes_of_path_coll)
        X = [[zeros(T,0)] for i in 1:M]
        N = [1,1]
        new{TV,T}(X, ρs, λs, sizes_of_path_coll, skip, N)
    end

    Adaptation{TV,T}() where {TV,T} = new{TV,T}()
end

NoAdaptation() = Adaptation{Val{false},Nothing}()

function still_adapting(adpt::Adaptation{Val{true}})
    adpt.N[1] > length(adpt.sizes) ? NoAdaptation() : adpt
end

still_adapting(adpt::Adaptation{Val{false}}) = adpt

function resize!(adpt::Adaptation{TV,T}, m, ns::Vector{Int64}) where {TV,T}
    K = length(adpt.X)
    for i in 1:K
        adpt.X[i] = [[zero(T) for _ in 1:ns[i]] for i in 1:m]
    end
end

function addPath!(adpt::Adaptation{Val{true},T}, X::Vector{SamplePath{T}}, i) where T
    if i % adpt.skip == 0
        m = length(X)
        for j in 1:m
            adpt.X[adpt.N[2]][j] .= X[j].yy
        end
    end
end

addPath!(adpt::Adaptation{Val{false}}, ::Any, ::Any) = false
