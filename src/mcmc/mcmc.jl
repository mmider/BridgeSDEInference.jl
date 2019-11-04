#===============================================================================
                                The main routine
===============================================================================#

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
#=

foo = MCMCSchedule(10, [[1,2,3],[3,4],[5,6,7,8]], (1,2,3,x->(x%5==0),true, x->(x%4==0)))


for (i,f) in enumerate(foo)
    if i == 5
        foo.updt_idx[1][1]=10000
    end
    print(f, "\n")
end
=#
