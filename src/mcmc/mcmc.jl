#===============================================================================
                                The main routine
===============================================================================#
"""
    mcmc(setup_mcmc::MCMCSetup, schedule::MCMCSchedule, setup)  <: ModelSetup

function for running the mcmc. receives as imput `MCMCSetup`, `MCMCschedule` and `setup`.
See [MCMCSetup](@ref), [MCMCSchedule](@ref). setup typically is DiffusionSetup making
additional setup choices when using the MCMC infrastructure to sample diffusion processes.
See [DiffusionSetup](@ref)
"""
function mcmc(setup_mcmc::MCMCSetup, schedule::MCMCSchedule, setup::T) where T <: ModelSetup
    ws, ll, θ = create_workspace(setup)
    ws_mcmc = create_workspace(setup_mcmc, schedule, θ)
    adpt = adaptation_object(setup, ws)

    aux = nothing
    for step in schedule
        step.save && save_imputed!(ws)
        for i in step.idx
            if step.param_updt || typeof(ws_mcmc.updates[i]) <: Imputation
                ws = next(ws, ws_mcmc.updates[i])
                ll, acc, θ = update!(ws_mcmc.updates[i], ws, θ, ll, step, aux)
                aux = aux_params(ws_mcmc.updates[i], aux)
                update!(ws_mcmc, acc, θ, i)
                step.verbose && print("\n")
            end
        end
        step.verbose && print("-----------------------------------------------",
                              "------\n")
        step.readjust && readjust!(ws_mcmc, step.iter)
        step.fuse && fuse!(ws_mcmc, schedule)
        ll = adaptation!(ws, adpt, step.iter, ll)
    end
    display_summary(ws, ws_mcmc)
    ws, ws_mcmc
end

display_summary(::Any, ::Any) = print("nothing to display...\n")
adaptation_object(::Any, ::Any) = nothing
adaptation!(::Any, ::Nothing, ::Any, ll) = ll
