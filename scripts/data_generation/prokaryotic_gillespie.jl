using DiffEqBiological

prokaryote_model = @reaction_network ProkaryoteRN begin
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

u0 = [8,8,8,5,5]
θ = [0.1, 0.7, 0.35, 0.2, 0.1, 0.9, 0.3, 0.1]
tt = (0.0, 100.0)


sol = simulate_data(u0, tt, θ)

using Plots; plot(sol)
