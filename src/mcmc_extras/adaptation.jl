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

check_if_adapt(::Adaptation{Val{T}}) where T = T

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
#=
addPath!(adpt::Adaptation{Val{false}}, ::Any, ::Any) = false








init_adaptation!(adpt::Adaptation{Val{false}}, 𝓦𝓢::Workspace) = nothing

function init_adaptation!(adpt::Adaptation{Val{true}}, 𝓦𝓢::Workspace)
    m = length(𝓦𝓢.XX)
    resize!(adpt, m, [length(𝓦𝓢.XX[i]) for i in 1:m])
end

function adaptationUpdt!(adpt::Adaptation{Val{false}}, 𝓦𝓢::Workspace, yPr, i,
                         ll, ::ObsScheme, ::ST) where {ObsScheme,ST}
    adpt, 𝓦𝓢, yPr, ll
end

function adaptationUpdt!(adpt::Adaptation{Val{true}}, 𝓦𝓢::Workspace, yPr, i,
                         ll, ::ObsScheme, ::ST) where {ObsScheme,ST}
    if i % adpt.skip == 0
        if adpt.N[2] == adpt.sizes[adpt.N[1]]
            X̄ = compute_X̄(adpt)
            m = length(𝓦𝓢.P)
            for j in 1:m
                Pt = recentre(𝓦𝓢.P[j].Pt, 𝓦𝓢.XX[j].tt, X̄[j])
                update_λ!(Pt, adpt.λs[adpt.N[1]])
                𝓦𝓢.P[j] = GuidPropBridge(𝓦𝓢.P[j], Pt)

                Ptᵒ = recentre(𝓦𝓢.Pᵒ[j].Pt, 𝓦𝓢.XX[j].tt, X̄[j])
                update_λ!(Ptᵒ, adpt.λs[adpt.N[1]])
                𝓦𝓢.Pᵒ[j] = GuidPropBridge(𝓦𝓢.Pᵒ[j], Ptᵒ)
            end
            𝓦𝓢 = Workspace(𝓦𝓢, adpt.ρs[adpt.N[1]])

            solveBackRec!(NoBlocking(), 𝓦𝓢.P, ST())
            #solveBackRec!(NoBlocking(), 𝓦𝓢.Pᵒ, ST())
            y = 𝓦𝓢.XX[1].yy[1]
            yPr = invStartPt(y, yPr, 𝓦𝓢.P[1])

            for j in 1:m
                invSolve!(Euler(), 𝓦𝓢.XX[j], 𝓦𝓢.WW[j], 𝓦𝓢.P[j])
            end
            ll = logpdf(yPr, y)
            ll += pathLogLikhd(ObsScheme(), 𝓦𝓢.XX, 𝓦𝓢.P, 1:m, 𝓦𝓢.fpt)
            ll += lobslikelihood(𝓦𝓢.P[1], y)
            adpt.N[2] = 1
            adpt.N[1] += 1
        else
            adpt.N[2] += 1
        end
    end
    adpt, 𝓦𝓢, yPr, ll
end

function compute_X̄(adpt::Adaptation{Val{true}})
    X = adpt.X
    num_paths = adpt.sizes[adpt.N[1]]
    num_segments = length(X[1])
    for i in 2:num_paths
        for j in 1:num_segments
            num_pts = length(X[i][j])
            for k in 1:num_pts
                X[1][j][k] += X[i][j][k]
            end
        end
    end
    for j in 1:num_segments
        num_pts = length(X[1][j])
        for k in 1:num_pts
            X[1][j][k] /= num_paths
        end
    end
    X[1]
end

print_adaptation_info(adpt::Adaptation{Val{false}}, ::Any, ::Any, ::Any) = nothing

function print_adaptation_info(adpt::Adaptation{Val{true}}, accImpCounter,
                               accUpdtCounter, i)
    if i % adpt.skip == 0 && adpt.N[2] == adpt.sizes[adpt.N[1]]
        print("--------------------------------------------------------\n")
        print(" Adapting...\n")
        print(" Using ", adpt.N[2], " many paths, thinned by ", adpt.skip, "\n")
        print(" Previous imputation acceptance rate: ", accImpCounter/i, "\n")
        print(" Previous param update acceptance rate: ", accUpdtCounter./i, "\n")
        print("--------------------------------------------------------\n")
    end
end
=#
