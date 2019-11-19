using DiffEqBiological
using Random
using Distributions

prokaryote_model = @reaction_network ProkaryoteRN begin
    # order of appearnce: DNA, P₂, DNAxP₂, RNA, P
    θ1, DNA + P₂ --> DNAxP₂
    θ2, DNAxP₂ --> DNA + P₂
    θ3, DNA --> DNA + RNA
    θ4, RNA --> RNA + P
    θ5, 2P --> P₂
    θ6, P₂ --> 2*P
    θ7, RNA --> 0*RNA
    θ8, P --> 0*P
end θ1 θ2 θ3 θ4 θ5 θ6 θ7 θ8


function simulate_data(u0, tt, rates)
    prob = DiscreteProblem(u0, tt, rates)
    jump_prob = JumpProblem(prob, Direct(), prokaryote_model)
    sol = solve(jump_prob, SSAStepper())
    sol
end

u0 = [5,8,5,8,8]
θ = [0.1, 0.7, 0.35, 0.2, 0.1, 0.9, 0.3, 0.1]
tt = (0.0, 300.0)

Random.seed!(4)
sol = simulate_data(u0, tt, θ)
obs_time = collect(tt[1]:1:tt[end])
# permute to obtain (RNA, P, P₂, DNA, DNAxP₂) ordering
XX = map(x->x[[4,5,2,1,3]], sol.(obs_time))

# Let's add noise
#obs = [o[2] + 2*o[3] for o in XX[2:end]] .+ rand(Normal(0.0, 2.0), length(obs)-1)
# Alternative
obs = [[o[1] + rand(Normal(0.0, 1.0)), o[2] + 2*o[3]+rand(Normal(0.0, 2.0))] for o in XX[2:end]]

# save the data
OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output7")

# save the process
open(joinpath(OUT_DIR, "prokaryote_custom.dat"), "w") do f
    write(f, "time RND P P2 DNA DNAxP2\n")
    for i in 1:length(obs_time)
        data = join(obs[i], " ")
        line = join([obs_time[i], " ", data, "\n"])
        write(f, line)
    end
end

# save just partial and noisy Observations
open(joinpath(OUT_DIR, "prokaryote_custom.dat"), "w") do f
    write(f, "time P+2*P2\n")
    for i in 2:length(obs_time)
        data = join(obs[i-1], " ")
        line = join([obs_time[i], " ", data, "\n"])
        write(f, line)
    end
end


# alternative save
open(joinpath(OUT_DIR, "prokaryote_custom.dat"), "w") do f
    write(f, "time RNA P+2*P2\n")
    for i in 2:length(obs_time)
        data = join(obs[i-1], " ")
        line = join([obs_time[i], " ", data, "\n"])
        write(f, line)
    end
end
