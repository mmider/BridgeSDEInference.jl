
mutable struct MCMCSetup{ObsScheme}
    P˟
    P̃
    blocking
    blocking_params
    setup_completion
    Ls
    Σs
    obs
    obs_times
    fpt
    dt
    τ
    t_kernel
    ρ
    param_updt
    updt_coord
    updt_type
    adaptive_prop
    priors
    x0_prior
    num_mcmc_steps
    save_iter
    verb_iter
    skip_for_save
    warm_up
    solver
    change_pt
    Wnr
    XX
    WW

    function MCMCSetup(P˟, P̃, ::ObsScheme) where ObsScheme <: AbstractObsScheme
        setup_completion = Dict(:obs => false,
                                :imput => false,
                                :tkern => false,
                                :prior => false,
                                :mcmc => false,
                                :solv => false)
        new{ObsScheme}(P˟, P̃, NoBlocking(), ([], 0.1, NoChangePt()),
                       setup_completion)
    end
end

function set_observations!(setup::MCMCSetup, Ls, Σs, obs, obs_times,
                           fpt=fill(nothing, length(obs)-1))
    setup.Ls = Ls
    setup.Σs = Σs
    setup.obs = obs
    setup.obs_times = obs_times
    setup.fpt = fpt
    setup.setup_completion[:obs] = true
end

function incomplete_message(::Val{:obs})
    print("\nThe observations have not been set up. Please call ",
          "set_observations!() passing the data on observations and the type ",
          "of observation scheme.\n")
end

function set_imputation_grid!(setup::MCMCSetup, dt, time_transf)
    setup.dt = dt
    setup.τ = time_transf
    setup.setup_completion[:imput] = true
end

function incomplete_message(::Val{:imput})
    print("\nThe imputation grid has not been set up. Please call ",
          "set_imputation_grid!() passing the `delta-t` and the time "
          "transformation.\n")
end

function set_transition_kernels!(setup::MCMCSetup, transition_kernel,
                                 crank_nicolson_memory=0.0, param_updt=true,
                                 updt_coord=(Val((true,)),),
                                 updt_type=(MetropolisHastingsUpdt(),),
                                 adaptive_proposals=NoAdaptation())
    setup.t_kernel = transition_kernel
    setup.ρ = crank_nicolson_memory
    setup.param_updt = param_updt
    setup.updt_coord = updt_coord
    setup.updt_type = updt_type
    setup.adaptive_proposals = adaptive_proposals
    setup.setup_completion[:tkern] = true
end

function incomplete_message(::Val{:tkern})
    print("\nThe transition kernels have not been set up. Please call "
          "set_transition_kernels!() passing the information about transition "
          "kernels for the parameters updates and path imputation.\n")
end

function set_priors!(setup::MCMCSetup, priors, x0_prior)
    setup.priors = priors
    setup.x0_prior = x0_prior
    setup.setup_completion[:prior] = true
end

function incomplete_message(::Val{:prior})
    print("\nThe priors have not been set up. Please call set_priors!() "
          "passing the priors on the parameters and on the starting point.\n")
end

function set_mcmc_params!(setup::MCMCSetup, num_mcmc_steps, save_iter=NaN,
                          verb_iter=NaN, skip_for_save=1, warm_up=0)
    setup.num_mcmc_steps = num_mcmc_steps
    setup.save_iter = save_iter
    setup.verb_iter = verb_iter
    setup.skip_for_save = skip_for_save
    setup.warm_up = warm_up
    setup.setup_completion[:mcmc] = true
end

function incomplete_message(::Val{:mcmc})
    print("\nThe parameters for MCMC have not been set up. Please call "
          "set_mcmc_params!() passing the auxiliary parameters required "
          "to set up the mcmc chain.\n")
end

function set_blocking!(setup::MCMCSetup, blocking::Blocking=NoBlocking(),
                       blocking_params=([], 0.1, NoChangePt()))
    setup.blocking = blocking
    setup.blocking_params = blocking_params
end

function set_solver!(setup::MCMCSetup, solver=Ralston3(),
                     change_pt=NoChangePt())
    setup.solver = solver
    setup.change_pt = change_pt
    setup.setup_completion[:solv] = true
end

function incomplete_message(::Val{:solv})
    print("\nThe types of solvers have not been determined. Please call "
          "set_solver!() to choose the ODE solver.\n")
end

function check_if_complete(setup::MCMCSetup,
                           i_range=keys(setup.setup_completion))
    complete = true
    for i in i_range
        if !setup.setup_completion[i]
            incomplete_message(Val(i))
            complete = false
        end
    end
    complete
end

function determine_data_type(setup::MCMCSetup)
    check_if_complete(setup, [:prior]) || throw(UndefRefError())
    x = start_pt(x0_prior)
    drift = b(0.0, x, setup.P˟)
    vola = σ(0.0, x, setup.P˟)
    # @assert typeof(x) == typeof(drift) # maybe this assertion is too strong
    typeof(drift), typeof(vola' * drift)
end

function prepare_containers!(setup::MCMCSetup)
    T, S = determine_data_type(setup)
    Wnr = Wiener{S}()
    TW = typeof(sample([0], Wnr))
    TX = typeof(SamplePath([], zeros(T, 0)))

    # TODO modify so that it will work with GPUarrays
    m = length(setup.obs)-1
    WW = Vector{TW}(undef,m)
    XX = Vector{TX}(undef,m)

    setup.Wnr = Wnr
    setup.WW = WW
    setup.XX = XX

    prepare_obs_containers!(T, setup)
end

function prepare_obs_containers!(::Type{T}, setup::MCMCSetup) where T <: Number
    setup.Σs = correct_data_type(setup.Σs, T)
    setup.Ls = correct_data_type(setup.Ls, T)
    setup.obs = correct_data_type(setup.obs, T)
end

function prepare_obs_containers!(::Type{T}, setup::MCMCSetup) where T <: Array
    setup.Σs = correct_data_type(setup.Σs, Array)
    setup.Ls = correct_data_type(setup.Ls, Array)
    setup.obs = correct_data_type(setup.obs, Array)
end

function prepare_obs_containers!(::Type{T}, setup::MCMCSetup) where T <:SArray
    f(x) = SMatrix{_dim(x)...}(x)
    setup.Σs = correct_data_type(setup.Σs, f)
    setup.Ls = correct_data_type(setup.Ls, f)
    setup.obs = correct_data_type(setup.obs, f)
end

function correct_data_type(vals, T)
    [T(val) for val in vals]
end

_dim(mat::T) where T <: Number = 1, 1

_dim(mat::T) where T <: Union{Array, SArray} = _dim(mat, Val{ndims(mat)}())

_dim(mat, ::Val{1}) = size(mat)[1], 1

_dim(mat, ::Val{2}) = size(mat)

_dim(mat, ::T) where T = throw(ArgumentError())



"""
    findProposalLaw(xx, tt, P˟, P̃, Ls, Σs; dt=1/5000, timeChange=true,
                    solver::ST=Ralston3())

Initialise the object with proposal law and all the necessary containers needed
for the simulation of the guided proposals
"""
function find_proposal_law!(::Type{K}, setup::MCMCSetup) where K
    xx, tt, P˟, P̃ = setup.obs, setup.obs_times, setup.P˟, setup.P̃
    Ls, Σs, solver = setup.Ls, setup.Σs, setup.solver
    change_pt = typeof(setup.change_pt)(setup.blocking_params[3])

    m = length(xx)-1
    P = Array{ContinuousTimeProcess,1}(undef,m)

    for i in m:-1:1
        num_pts = Int64(ceil((tt[i+1]-tt[i])/dt))+1
        t = τ(tt[i], tt[i+1]).( range(tt[i], stop=tt[i+1], length=num_pts) )
        P[i] = ( (i==m) ? GuidPropBridge(K, t, P˟, P̃[i], Ls[i], xx[i+1], Σs[i];
                                         change_pt=change_pt, solver=solver) :
                          GuidPropBridge(K, t, P˟, P̃[i], Ls[i], xx[i+1], Σs[i],
                                         P[i+1].H[1], P[i+1].Hν[1], P[i+1].c[1];
                                         change_pt=change_pt, solver=solver) )
    end
    setup.P = P
end


function initialise(setup::MCMCSetup; verbose=false)
    verbose && print("Initialising MCMC setup...\nPreparing containers...\n")
    prepare_containers!(setup)
    verbose && print("Initialising proposal laws...\n")
    find_proposal_law!(#TODO add variable,
                       setup)
    #TODO initialise for computation of gradients
end

#NOTE code to adapt:
#θ = params(P[1].Target)
#ϑs = [[θ[j] for j in idx(uc)] for uc in updtCoord]
#result = [DiffResults.GradientResult(ϑ) for ϑ in ϑs]
#resultᵒ = [DiffResults.GradientResult(ϑ) for ϑ in ϑs]
#Q = eltype(result)
