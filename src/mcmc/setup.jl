#=
    -------------------------------------------------------------------------
    Implements functionalities for setting up the Markov chain Monte Carlo
    algorithm. The main object is `MCMCSetup` and its members comprise of
    all objects that need to be passed to the `mcmc` function from `mcmc.jl`.
    The remaining routines are used to populate instances of `MCMCSetup` and
    verify its fields in a structured manner.
    --------------------------------------------------------------------------
=#


"""
    MCMCSetup

Groups together all objects that need to be passed to `mcmc` function from
`mcmc.jl`.
"""
mutable struct MCMCSetup{ObsScheme}
    P˟                  # Target diffusion law
    P̃                   # Vector of auxiliary diffusion laws
    blocking            # Blocking type
    blocking_params     # Parameters for blocking
    setup_completion    # (internal) Indicates progress of setting-up MCMCSetup
    Ls                  # Vector of observation operators
    Σs                  # Vector of covariance matrices
    obs                 # Observations
    obs_times           # Recorded times of observations
    fpt                 # Vector with information about first-passage times
    dt                  # Granularity of the imputation grid
    τ                   # Time transformation for the imputation grid
    #TODO remember to change to collection
    t_kernel            # Collection of transition kernels for param updt step
    ρ                   # Memory parameter of the precond Crank-Nicolson scheme
    param_updt          # Flag for whether to update parameters at all
    updt_coord          # Collection indicating which coordinates to update
    updt_type           # Collection indicating types of parameter updates
    adaptive_prop       # Object for adapting guided proposals
    priors              # Priors over parameters
    x0_prior            # Prior over the starting position
    x0_guess            # Guess for a position of a starting point
    num_mcmc_steps      # Total number of steps of the mcmc sampler
    save_iter           # Save the path every ... iteration
    verb_iter           # Print progress message to console every ... iteration
    skip_for_save       # When saving path, thin the grid by a factor of ...
    warm_up             # Number of steps of the chain in which no param update is made
    solver              # Type of ODE solver
    change_pt           # Change point between ODEs for M,L,μ and H,Hν,c
    Wnr                 # Definition of the driving Wiener law
    XX                  # Container for the path
    WW                  # Container for the driving noise
    P                   # Container for the guided proposals

    """
        MCMCSetup(P˟, P̃, ::ObsScheme)

    Initialise `MCMCSetup` with a given target law `P˟`, auxiliary laws `P̃` and
    observation scheme `ObsScheme`.
    """
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


"""
    set_observations!(setup::MCMCSetup, Ls, Σs, obs, obs_times,
                      fpt=fill(nothing, length(obs)-1))

Store observations to `setup`. The observations follow the scheme `V=LX+η`,
where V are the observations in `obs`, observed at times given in `obs_times`,
`L` are the observation operators given in `Ls`, `X` is the unobserved,
underlying diffusion and `η` is a Gaussian noise with mean `0` and covariance
`Σ` with the last stored in `Σs`. `fpt` provides additional information in case
the nature of observations has to do with first-passage times
"""
function set_observations!(setup::MCMCSetup, Ls, Σs, obs, obs_times,
                           fpt=fill(nothing, length(obs)-1))
    setup.Ls = Ls
    setup.Σs = Σs
    setup.obs = obs
    setup.obs_times = obs_times
    setup.fpt = fpt
    setup.setup_completion[:obs] = true
end


"""
    incomplete_message(::Val{:obs})

Print message to a console for incomplete step of setting observations
"""
function incomplete_message(::Val{:obs})
    print("\nThe observations have not been set up. Please call ",
          "set_observations!() passing the data on observations and the type ",
          "of observation scheme.\n")
end


"""
    set_imputation_grid!(setup::MCMCSetup, dt,
                         time_transf=(t₀,T) -> ((x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))))

Define the imputation grid in `setup`. `dt` defines the granulatrity of the
imputation grid and `time_transf` defines a time transformation to use for
transforming equidistant grid.
"""
function set_imputation_grid!(setup::MCMCSetup, dt,
                              time_transf=(t₀,T) -> ((x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))))
    setup.dt = dt
    setup.τ = time_transf
    setup.setup_completion[:imput] = true
end

"""
    incomplete_message(::Val{:imput})

Print message to a console for incomplete step of setting imputation grid
"""
function incomplete_message(::Val{:imput})
    print("\nThe imputation grid has not been set up. Please call ",
          "set_imputation_grid!() passing the `delta-t` and the time ",
          "transformation.\n")
end


"""
    set_transition_kernels!(setup::MCMCSetup, transition_kernel,
                            crank_nicolson_memory=0.0, param_updt::Bool=true,
                            updt_coord=(Val((true,)),),
                            updt_type=(MetropolisHastingsUpdt(),),
                            adaptive_proposals=NoAdaptation())

Store the transition kernels for parameter update steps as well path imputation
step in the `setup` object.
...
# Arguments
- `setup`: object to be set up
- `transition_kernel`: collection of transition kernels (one for each param updt)
- `crank_nicolson_memory`: memory parameter for random walk on a path space
- `param_updt`: flag for whether to update parameters at all
- `updt_coord`: collection indicating which coordinates to update
- `updt_type`: collection indicating types of parameter updates
- `adaptive_proposals`: object for adapting guided proposals
...
"""
function set_transition_kernels!(setup::MCMCSetup, transition_kernel,
                                 crank_nicolson_memory=0.0,
                                 param_updt::Bool=false,
                                 updt_coord=nothing,
                                 updt_type=(MetropolisHastingsUpdt(),),
                                 adaptive_proposals=NoAdaptation())
    setup.t_kernel = transition_kernel
    setup.ρ = crank_nicolson_memory
    setup.param_updt = param_updt
    setup.updt_coord = reformat_updt_coord(updt_coord, params(setup.P˟))
    setup.updt_type = updt_type
    @assert length(setup.updt_type) == length(setup.updt_coord)
    setup.adaptive_prop = adaptive_proposals
    setup.setup_completion[:tkern] = true
end

"""
    reformat_updt_coord(updt_coord::Nothing, θ)

Chosen not to update parameters, returned object is not important
"""
reformat_updt_coord(updt_coord::Nothing, θ) = (Val((true,)),)


IntContainer = Union{Number,NTuple{N,<:Integer},Vector{<:Integer}} where N
IntLongContainer = Union{NTuple{N,<:Integer},Vector{<:Integer}} where N
DoubleContainer = Union{Vector{<:IntLongContainer},NTuple{N,Tuple}} where N
"""
    reformat_updt_coord(updt_coord::S, θ) where S<:IntContainer

Single joint update of multiple parameters at once
"""
function reformat_updt_coord(updt_coord::S, θ) where S<:IntContainer
    (int_to_val(updt_coord, θ),)
end

"""
    reformat_updt_coord(updt_coord::S, θ) where S<:DoubleContainer

Multiple updates, reformat from indices to update to a tuple of Val{...}()
"""
function reformat_updt_coord(updt_coord::S, θ) where S<:DoubleContainer
    Tuple([int_to_val(uc, θ) for uc in updt_coord])
end

"""
    int_to_val(updt_coord::S, θ)

Transform from a container of indices to update to a
Val{tuple of true/false one-hot-encoding}()
"""
function int_to_val(updt_coord::S, θ) where S<:IntContainer
    @assert all([1 <= uc <= length(θ) for uc in updt_coord])
    Val{Tuple([i in updt_coord for i in 1:length(θ)])}()
end


"""
    reformat_updt_coord(updt_coord::Nothing, θ)

If the user does not use indices of coordinates to be updated it is assumed that
appropriate Val{(...)}() object is passed and nothing is done, use at your own risk
"""
reformat_updt_coord(updt_coord, θ) = updt_coord

"""
    incomplete_message(::Val{:tkern})

Print message to a console for incomplete step of setting transition kernels
"""
function incomplete_message(::Val{:tkern})
    print("\nThe transition kernels have not been set up. Please call ",
          "set_transition_kernels!() passing the information about transition ",
          "kernels for the parameters updates and path imputation.\n")
end

"""
    set_priors!(setup::MCMCSetup, priors, x0_prior, x0_guess=nothing)

Store the priors over parameters (in `priors`) and over the starting point (in
`x0_prior`) into the object `setup`.
"""
function set_priors!(setup::MCMCSetup, priors, x0_prior, x0_guess=nothing)
    setup.priors = priors
    setup.x0_prior = x0_prior
    setup.setup_completion[:prior] = true

    if x0_guess == nothing
        @assert typeof(x0_prior) <: KnownStartingPt
        x0_guess = start_pt(nothing, x0_prior)
    end
    setup.x0_guess = x0_guess
end

"""
    incomplete_message(::Val{:prior})

Print message to a console for incomplete step of setting priors
"""
function incomplete_message(::Val{:prior})
    print("\nThe priors have not been set up. Please call set_priors!() ",
          "passing the priors on the parameters and on the starting point.\n")
end

"""
    set_mcmc_params!(setup::MCMCSetup, num_mcmc_steps, save_iter=NaN,
                     verb_iter=NaN, skip_for_save=1, warm_up=0)

Define the parametrisation of the mcmc sampler.
...
# Arguments
- `setup`: object to be set up
- `num_mcmc_steps`: total number of steps of the mcmc sampler
- `save_iter`: save the path every ... iteration
- `verb_iter`: print progress message to console every ... iteration
- `skip_for_save`: when saving path, thin the grid by a factor of ...
- `uwarm_up`: number of steps of the chain in which no param update is made
...
"""
function set_mcmc_params!(setup::MCMCSetup, num_mcmc_steps, save_iter=NaN,
                          verb_iter=NaN, skip_for_save=1, warm_up=0)
    setup.num_mcmc_steps = num_mcmc_steps
    setup.save_iter = save_iter
    setup.verb_iter = verb_iter
    setup.skip_for_save = skip_for_save
    setup.warm_up = warm_up
    setup.setup_completion[:mcmc] = true
end

"""
    incomplete_message(::Val{:mcmc})

Print message to a console for incomplete step of setting mcmc parametrisation
"""
function incomplete_message(::Val{:mcmc})
    print("\nThe parameters for MCMC have not been set up. Please call ",
          "set_mcmc_params!() passing the auxiliary parameters required ",
          "to set up the mcmc chain.\n")
end

"""
    set_blocking!(setup::MCMCSetup, blocking::Blocking=NoBlocking(),
                  blocking_params=([], 0.1, NoChangePt()))

Store information about blocking in `setup`. `blocking` indicates the type of
blocking and `blocking_params` passes additional necessary parameters
"""
function set_blocking!(setup::MCMCSetup, blocking::Blocking=NoBlocking(),
                       blocking_params=([], 0.1, NoChangePt())
                       ) where {Blocking <: BlockingSchedule}
    setup.blocking = blocking
    setup.blocking_params = blocking_params
end

"""
    set_solver!(setup::MCMCSetup, solver=Ralston3(), change_pt=NoChangePt())

Define the ODE solvers in `setup`. `solver` defines the type of ODE solver and
`change_pt` gives information about the change point for switching between the
solvers for M,L,μ and H,Hν,c
"""
function set_solver!(setup::MCMCSetup, solver=Ralston3(),
                     change_pt=NoChangePt())
    setup.solver = solver
    setup.change_pt = change_pt
    setup.setup_completion[:solv] = true
end

"""
    incomplete_message(::Val{:solv})

Print message to a console for incomplete step of setting ODE solvers
"""
function incomplete_message(::Val{:solv})
    print("\nThe types of solvers have not been determined. Please call ",
          "set_solver!() to choose the ODE solver.\n")
end

"""
    check_if_complete(setup::MCMCSetup, labels=keys(setup.setup_completion))

Check if all the set-up steps listed in `labels` have been finalised in `setup`
"""
function check_if_complete(setup::MCMCSetup,
                           labels=keys(setup.setup_completion))
    complete = true
    for label in labels
        if !setup.setup_completion[label]
            incomplete_message(Val(label))
            complete = false
        end
    end
    complete
end

"""
    determine_data_type(setup::MCMCSetup)

Determine the data type of the containers with path and driving noise
"""
function determine_data_type(setup::MCMCSetup)
    check_if_complete(setup, [:prior]) || throw(UndefRefError())
    x = setup.x0_guess
    drift = b(0.0, x, setup.P˟)
    vola = σ(0.0, x, setup.P˟)
    # @assert typeof(x) == typeof(drift) # maybe this assertion is too strong
    typeof(drift), typeof(vola' * drift)
end

"""
    prepare_containers!(setup::MCMCSetup)

Set-up the containers for paths, driving noise, observations, observation noise
and observation operators, automatically choosing an appropriate data type based
on the return values of the drift and volatility coefficients
"""
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

"""
    prepare_obs_containers!(::Type{T}, setup::MCMCSetup)

Define containers for the observations, observation noise and observation
operators based on the passed data-type
"""
function prepare_obs_containers!(::Type{T}, setup::MCMCSetup) where T <: Number
    correct_data_type!(setup, T, T)
end

"""
    prepare_obs_containers!(::Type{T}, setup::MCMCSetup)

Define containers for the observations, observation noise and observation
operators based on the passed data-type
"""
function prepare_obs_containers!(::Type{T}, setup::MCMCSetup) where T <: Array
    correct_data_type!(setup, Array, Array)
end

"""
    prepare_obs_containers!(::Type{T}, setup::MCMCSetup)

Define containers for the observations, observation noise and observation
operators based on the passed data-type
"""
function prepare_obs_containers!(::Type{T}, setup::MCMCSetup) where T <:SArray
    f(x) = SMatrix{_dim(x)...}(x)

    g(x, ::Val{1}=Val{ndims(x)}()) = SVector{size(x)...}(x)
    g(x, ::Val{2}=Val{ndims(x)}()) = SMatrix{size(x)...}(x)

    correct_data_type!(setup, f, g)
end

"""
    correct_data_type!(setup::MCMCSetup, f, g)

Transform the elements of observations, observation noise and observation
operators collections stored in `setup` according to functions `f` and `g`
"""
function correct_data_type!(setup::MCMCSetup, f, g)
    setup.Σs = map(f, setup.Σs)
    setup.Ls = map(f, setup.Ls)
    setup.obs = map(g, setup.obs)
end

"""
    _dim(mat::T) where T <: Number

Size of a matrix corresponding to a scalar is (1,1)
"""
_dim(mat::T) where T <: Number = 1, 1

"""
    _dim(mat::T) where T <: Union{Array, SArray}

Size of a matrix corresponding to an element `mat`
"""
_dim(mat::T) where T <: Union{Array, SArray} = _dim(mat, Val{ndims(mat)}())


"""
    _dim(mat, ::Val{1})
Size of a matrix corresponding to a vector is (vector length, 1)
"""
_dim(mat, ::Val{1}) = size(mat)[1], 1

"""
    _dim(mat, ::Val{1})
Size of a matrix corresponding to a matrix `mat` is just size(mat)
"""
_dim(mat, ::Val{2}) = size(mat)

"""
    _dim(mat, ::T) where T
Size of a matrix corresponding to tensor with dimension larger than matrix is
undefined
"""
_dim(mat, ::T) where T = throw(ArgumentError())


function _build_time_grid(τ, dt, t0, T)
    num_pts = Int64(ceil((T-t0)/dt))+1
    tt = τ(t0, T).( range(t0, stop=T, length=num_pts) )
    tt
end

 #TODO remove this K, it's most likely not needed
"""
    find_proposal_law!(::Type{K}, setup::MCMCSetup) where K

Initialise the object with proposal law and all the necessary containers needed
for the simulation of the guided proposals
"""
function find_proposal_law!(::Type{K}, setup::MCMCSetup) where K
    xx, tt, P˟, P̃ = setup.obs, setup.obs_times, setup.P˟, setup.P̃
    Ls, Σs, solver, dt, τ = setup.Ls, setup.Σs, setup.solver, setup.dt, setup.τ
    change_pt = typeof(setup.change_pt)(get_change_pt(setup.blocking_params[3]))

    m = length(xx)-1
    t = _build_time_grid(τ, dt, tt[m], tt[m+1])
    P = GuidPropBridge(K, t, P˟, P̃[m], Ls[m], xx[m+1], Σs[m];
                       change_pt=change_pt, solver=solver)
    #P = Array{ContinuousTimeProcess,1}(undef,m)
    Ps = [deepcopy(P)]

    for i in m-1:-1:1
        t = _build_time_grid(τ, dt, tt[i], tt[i+1])
        P = GuidPropBridge(K, t, P˟, P̃[i], Ls[i], xx[i+1], Σs[i], P.H[1],
                           P.Hν[1], P.c[1]; change_pt=change_pt, solver=solver)
        push!(Ps, deepcopy(P))
    end
    setup.P = reverse(Ps)
end

"""
    initialise(::Type{K}, setup::MCMCSetup; verbose=false)
Initialise the internal containers of `setup`. Check if all the necessary data
has been passed to `setup`
"""
function initialise!(::Type{K}, setup::MCMCSetup, verbose=false) where K
    verbose && print("Initialising MCMC setup...\nPreparing containers...\n")
    prepare_containers!(setup)
    verbose && print("Initialising proposal laws...\n")
    find_proposal_law!(K, setup) #TODO remove this K, it's most likely not needed
    #TODO initialise for computation of gradients
end

"""
    check_if_recompute_ODEs(setup::MCMCSetup)

Utility function for checking if H,Hν,c need to be re-computed for a respective
parameter update
"""
function check_if_recompute_ODEs(setup::MCMCSetup)
    [any([e in depends_on_params(setup.P[1].Pt) for e
          in idx(uc)]) for uc in setup.updt_coord]
end


#NOTE code to adapt:
#θ = params(P[1].Target)
#ϑs = [[θ[j] for j in idx(uc)] for uc in updtCoord]
#result = [DiffResults.GradientResult(ϑ) for ϑ in ϑs]
#resultᵒ = [DiffResults.GradientResult(ϑ) for ϑ in ϑs]
#Q = eltype(result)
