# Test that the transition density
# in a non-linear, non-homogenous, non-constant diffusivity model
# estimated by forward simulation
# agrees with the density obtained from the linearisation
# reweighted with importance weights using guided proposals.
const 𝕏 = SVector{1}
using GaussianDistributions

struct TargetSDE <: Bridge.ContinuousTimeProcess{Float64}
end
struct LinearSDE{T}  <: Bridge.ContinuousTimeProcess{Float64}
    σ::T
end
Bridge.b(s, x, P::TargetSDE) = -0.1x + .5sin(x[1]) + 0.5sin(s/4)
Bridge.b(s, x, P::LinearSDE) = Bridge.B(s, P)*x + Bridge.β(s, P)
Bridge.B(s, P::LinearSDE) = SMatrix{1}(-0.1)
Bridge.β(s, P::LinearSDE) = 𝕏(0.5sin(s/4))

Bridge.σ(s, x, P::TargetSDE) = SMatrix{1}(2.0 + 0.5cos(x[1]))
Bridge.σ(s, x, P::LinearSDE) = P.σ
Bridge.σ(s, P::LinearSDE) = P.σ
Bridge.a(s, P::LinearSDE) = P.σ^2

Bridge.constdiff(::TargetSDE) = false
Bridge.constdiff(::LinearSDE) = true

binind(r, x) = searchsortedfirst(r, x) - 1


function test_measchange()
    Random.seed!(1)
    fextra = 1.
    f = 1.0
    T = round(f*4*pi, digits=2)
    P = TargetSDE()
    v = 𝕏(pi/2)

    x0 = 𝕏(-pi/2)

    t = 0:0.01*f*fextra:T
    t = Bridge.tofs.(t, 0, T)
    W = Bridge.samplepath(t, 𝕏(0.0))

    Wnr =  Wiener{𝕏{Float64}}()

    Σ = SMatrix{1}(0.1)
    L = SMatrix{1}(1.0)
    Noise = Gaussian(𝕏(0.0), Σ)

    sample!(W, Wnr)
    X = solve(Euler(), x0, W, P)
    v1 = X.yy[end]
    X.yy[end] = zero(v1)
    solve!(Euler(), X, x0, W, P)
    @test v1 ≈ X.yy[end]



    K = 50
    vrange = range(-10,10, length=K+1)
    vints = [(vrange[i], vrange[i+1]) for i in 1:K]

    k = 1
    N = 50000

    # Forward simulation

    vs = Float64[]
    for i in 1:N
        sample!(W, Wnr)
        solve!(Euler(), X, x0, W, P)
        v = L*X.yy[end] + rand(Noise)
        push!(vs, v[1])
    end

    counts = zeros(K+2)
    [counts[binind(vrange, v)+1] += 1 for v in vs]
    counts /= length(vs)

    wcounts = zeros(K)


    VProp = Uniform(-10,10)

    fpt = fill(NaN, 1)
    v = 𝕏(0.0)
    P̃ = LinearSDE(Bridge.σ(T, v, P)) # use a law with large variance
    Pᵒ = BSI.GuidPropBridge(eltype(x0), t, P, P̃, L, v, Σ)

    # Guided proposals

    for i in 1:N
        v = 𝕏(5*rand(VProp))
        while !((binind(vrange, v[1]) in 1:K))
            v = 𝕏(5*rand(VProp))
        end
        # other possibility: change proposal each step
    #    P̃ = LinearSDE(Bridge.σ(T, v, P))
        Pᵒ = BSI.GuidPropBridge(eltype(x0), t, P, P̃, L, v, Σ)

        sample!(W, Wnr)
        solve!(Euler(), X, x0, W, Pᵒ)
        ll = BSI.path_log_likhd(BSI.PartObs(), [X], [Pᵒ], 1:1, fpt, skipFPT=true)
        ll += BSI.lobslikelihood(Pᵒ, x0)
        ll -= logpdf(VProp, v[1])
        wcounts[binind(vrange, v[1])] += exp(ll)/N
    end
    bias = wcounts - counts[2:end-1]

    @testset "Statistical correctness of guided proposals" begin
        @test norm(bias) < 0.05
    end
end

test_measchange()
