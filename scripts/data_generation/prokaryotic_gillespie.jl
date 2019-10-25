using DiffEqBiological

#NOTE I think this will not work, because to all equations are of the
# Mass action reaction type.
function simulate_data_deprecated(rates, prob)
    reactant_stoich =
    [
        [4 => 1, 3 => 1],   # 1*DNA + 1*P₂
        [5 => 1],           # 1*DNA.P₂
        [4 => 1],           # 1*DNA
        [1 => 1],           # 1*RNA
        [2 => 1],           # 2*P
        [3 => 1],           # 1*P₂
        [1 => 1],           # 1*RNA
        [2 => 1]            # 1*P
    ]
    net_stoich =
    [
        [5 => 1],           # 1*DNA.P₂
        [4 => 1, 3 => 1],   # 1*DNA + 1*P₂
        [4 => 1, 1 => 1],   # 1*DNA + 1*RNA
        [1 => 1, 2 => 1],   # 1*RNA + 1*P
        [3 => 1],           # 1*P₂
        [2 => 2],           # 2*P
        [1 => 0],           # 0
        [2 => 0]            # 0
    ]
    mass_act_jump = MassActionJump(rates, reactant_stoich, net_stoich)
    jump_prob = JumpProblem(prob, Direct(), mass_act_jump)
    sol = solve(jump_prob, SSAStepper())
    sol
end


prokaryote_model = @reaction_network ProkaryoteRN begin
    θ1, DNA + P₂ --> DNAxP₂
    θ2, DNAxP₂ --> DNA + P₂
    θ3, DNA --> DNA + P₂
    θ4, RNA --> RNA + P
    θ5, 2P --> P₂
    θ6, P₂ --> 2*P₂
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


#NOTE this halts my departmental computer, pending repeat at home...
sol = simulate_data(u0, tt, θ)

using Plots; plot(sol)
