#=
    -------------------------------------------------------------------------
    Implements functionalities for setting up the Markov chain Monte Carlo
    algorithm. The main object is `MCMCSetup` and its members comprise of
    all objects that need to be passed to the `mcmc` function from `mcmc.jl`.
    The remaining routines are used to populate instances of `MCMCSetup` and
    verify its fields in a structured manner.
    --------------------------------------------------------------------------
=#

#===============================================================================
                                MCMC setup
===============================================================================#
"""
    mutable struct MCMCSetup

Implements functionalities for setting up the Markov chain Monte Carlo
algorithm. The main object is `MCMCSetup` and its members comprise of
all objects that need to be passed to the `mcmc` function from `mcmc.jl`.
The remaining routines are used to populate instances of `MCMCSetup` and
verify its fields in a structured manner.
"""
mutable struct MCMCSetup
    updates
    #readjust_params
    # add parameters for fusion, correlation matrix, historical acceptance rates
    function MCMCSetup(updates...)
        new([updates...])
    end
end

function set_readjust_params!(setup::MCMCSetup, readjust_params)
    setup.readjust_params = readjust_params
end




#===============================================================================
                                Diffusion setup
===============================================================================#

"""
    DiffusionSetup{ObsScheme} <: ModelSetup

Setup choices relevant to the path augmentation step to be passed to `mcmc` function from
`mcmc.jl`.


# Example:
```
         DiffusionSetup(P, P̃)
         DiffusionSetup(P, P̃, fptOrPartObs)
```
where P is the target process and P̃ is the linear approximation and possibly a vector of Booleans whether
the observations are first passage times.
"""
mutable struct DiffusionSetup{ObsScheme} <: ModelSetup
    setup_completion    # (internal) Indicates progress of setting-up DiffusionSetup
    P˟                  # Target diffusion law
    P̃                   # Vector of auxiliary diffusion laws
    adaptive_prop       # Object for adapting guided proposals
    skip_for_save       # When saving path, thin the grid by a factor of ...
    Ls                  # Vector of observation operators
    Σs                  # Vector of covariance matrices
    obs                 # Observations
    obs_times           # Recorded times of observations
    fpt                 # Vector with information about first-passage times
    dt                  # Granularity of the imputation grid
    τ                   # Time transformation for the imputation grid
    x0_prior            # Prior over the starting position
    x0_guess            # Guess for a position of a starting point
    Wnr                 # Definition of the driving Wiener law
    XX                  # Container for the path
    WW                  # Container for the driving noise
    P                   # Container for the guided proposals

    """
        DiffusionSetup(P˟, P̃, ::ObsScheme)

    Initialise `DiffusionSetup` with a given target law `P˟`, auxiliary laws `P̃` and
    observation scheme `ObsScheme`.
    """
    function DiffusionSetup(P˟, P̃, ::ObsScheme) where ObsScheme <: AbstractObsScheme
        new{ObsScheme}(Dict(:obs => false,
                            :imput => false,
                            :prior => false),
                       P˟, P̃, NoAdaptation(), 1)#NoBlocking(), ([], 0.1, NoChangePt()),
    end
end

adaptation_object(setup::DiffusionSetup) = deepcopy(setup.adaptive_prop)

"""
    set_observations!(setup::DiffusionSetup, Ls, Σs, obs, obs_times,
                      fpt=fill(nothing, length(obs)-1))

Store observations to `setup`. The observations follow the scheme `V=LX+η`,
where V are the observations in `obs`, observed at times given in `obs_times`,
`L` are the observation operators given in `Ls`, `X` is the unobserved,
underlying diffusion and `η` is a Gaussian noise with mean `0` and covariance
`Σ` with the last stored in `Σs`. `fpt` provides additional information in case
the nature of observations has to do with first-passage times
"""
function set_observations!(setup::DiffusionSetup, Ls, Σs, obs, obs_times,
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


"""
    set_imputation_grid!(setup::DiffusionSetup, dt,
                         time_transf=(t₀,T) -> ((x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))))

Define the imputation grid in `setup`. `dt` defines the granulatrity of the
imputation grid and `time_transf` defines a time transformation to use for
transforming equidistant grid.
"""
function set_imputation_grid!(setup::DiffusionSetup, dt,
                              time_transf=(t₀,T) -> ((x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))))
    setup.dt = dt
    setup.τ = time_transf
    setup.setup_completion[:imput] = true
end

function incomplete_message(::Val{:imput})
    print("\nThe imputation grid has not been set up. Please call ",
          "set_imputation_grid!() passing the `delta-t` and the time ",
          "transformation.\n")
end


"""
    set_x0_prior!(setup::DiffusionSetup, x0_prior, x0_guess=nothing)

Store the priors over the starting point into the object `setup`.
"""
function set_x0_prior!(setup::DiffusionSetup, x0_prior, x0_guess=nothing)
    setup.x0_prior = x0_prior
    if x0_guess == nothing
        @assert typeof(x0_prior) <: KnownStartingPt
        x0_guess = start_pt(nothing, x0_prior)
    end
    setup.x0_guess = x0_guess
    setup.setup_completion[:prior] = true
end

function incomplete_message(::Val{:prior})
    print("\nThe priors have not been set up. Please call set_priors!() ",
          "passing the priors on the parameters and on the starting point.\n")
end


function set_auxiliary!(setup::DiffusionSetup; skip_for_save=nothing,
                        adaptive_prop=nothing)
    if skip_for_save !== nothing
        setup.skip_for_save = skip_for_save
    end
    if adaptive_prop !== nothing
        setup.adaptive_prop = adaptive_prop
    end
end



"""
    check_if_complete(setup::DiffusionSetup, labels=keys(setup.setup_completion))

Check if all the set-up steps listed in `labels` have been finalised in `setup`
"""
function check_if_complete(setup::DiffusionSetup,
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
    determine_data_type(setup::DiffusionSetup)

Determine the data type of the containers with path and driving noise
"""
function determine_data_type(setup::DiffusionSetup)
    check_if_complete(setup, [:prior]) || throw(UndefRefError())
    x = setup.x0_guess
    drift = b(0.0, x, setup.P˟)
    vola = σ(0.0, x, setup.P˟)
    # @assert typeof(x) == typeof(drift) # maybe this assertion is too strong
    typeof(drift), typeof(vola' * drift)
end

"""
    prepare_containers!(setup::DiffusionSetup)

Set-up the containers for paths, driving noise, observations, observation noise
and observation operators, automatically choosing an appropriate data type based
on the return values of the drift and volatility coefficients
"""
function prepare_containers!(setup::DiffusionSetup)
    T, S = determine_data_type(setup)
    Wnr = Wiener{S}()
    TW = typeof(sample([0], Wnr))
    TX = typeof(SamplePath([], zeros(T, 0)))

    # TODO modify so that it will work with GPUArrays
    m = length(setup.obs)-1
    WW, XX = Vector{TW}(undef,m), Vector{TX}(undef,m)
    setup.Wnr, setup.WW, setup.XX = Wnr, WW, XX

    prepare_obs_containers!(T, setup)
end

"""
    prepare_obs_containers!(::Type{T}, setup::DiffusionSetup)

Define containers for the observations, observation noise and observation
operators based on the passed data-type
"""
function prepare_obs_containers!(::Type{T}, setup::DiffusionSetup) where T <: Number
    correct_data_type!(setup, T, T)
end

"""
    prepare_obs_containers!(::Type{T}, setup::DiffusionSetup)

Define containers for the observations, observation noise and observation
operators based on the passed data-type
"""
function prepare_obs_containers!(::Type{T}, setup::DiffusionSetup) where T <: Array
    correct_data_type!(setup, Array, Array)
end

"""
    prepare_obs_containers!(::Type{T}, setup::DiffusionSetup)

Define containers for the observations, observation noise and observation
operators based on the passed data-type
"""
function prepare_obs_containers!(::Type{T}, setup::DiffusionSetup) where T <:SArray
    f(x) = SMatrix{_dim(x)...}(x)

    g(x) = g(x, Val{ndims(x)}())
    g(x, ::Val{0}) = SVector{1}(x)
    g(x, ::Val{1}) = SVector{size(x)...}(x)
    g(x, ::Val{2}) = SMatrix{size(x)...}(x)

    correct_data_type!(setup, f, g)
end

"""
    correct_data_type!(setup::DiffusionSetup, f, g)

Transform the elements of observations, observation noise and observation
operators collections stored in `setup` according to functions `f` and `g`
"""
function correct_data_type!(setup::DiffusionSetup, f, g)
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
    find_proposal_law!(::Type{K}, setup::DiffusionSetup, solver, change_pt
                            ) where K

Initialise the object with proposal law and all the necessary containers needed
for the simulation of the guided proposals
"""
function find_proposal_law!(::Type{K}, setup::DiffusionSetup, solver, change_pt
                            ) where K
    xx, tt, P˟, P̃ = setup.obs, setup.obs_times, setup.P˟, setup.P̃
    Ls, Σs, dt, τ = setup.Ls, setup.Σs, setup.dt, setup.τ
    #NOTE remember to pre-allocate space for the M,L,mu containers for change_pt with blocking
    #change_pt = typeof(setup.change_pt)(get_change_pt(setup.blocking_params[3]))

    m = length(xx)-1
    t = _build_time_grid(τ, dt, tt[m], tt[m+1])
    P = GuidPropBridge(K, t, P˟, P̃[m], Ls[m], xx[m+1], Σs[m];
                       change_pt=change_pt, solver=solver)
    #P = Array{ContinuousTimeProcess,1}(undef,m)
    Ps = []
    push!(Ps, deepcopy(P))

    for i in m-1:-1:1
        t = _build_time_grid(τ, dt, tt[i], tt[i+1])
        P = GuidPropBridge(K, t, P˟, P̃[i], Ls[i], xx[i+1], Σs[i], P.H[1],
                           P.Hν[1], P.c[1]; change_pt=change_pt, solver=solver)
        push!(Ps, deepcopy(P))
    end
    Ps = [P for P in Ps]
    setup.P = reverse(Ps)
end

"""
    initialise(::Type{K}, setup::DiffusionSetup; verbose=false)
Initialise the internal containers of `setup`. Check if all the necessary data
has been passed to `setup`
"""
function initialise!(::Type{K}, setup::DiffusionSetup, solver, verbose=false,
                     change_pt=NoChangePt()) where K
    verbose && print("Initialising MCMC setup...\nPreparing containers...\n")
    prepare_containers!(setup)
    verbose && print("Initialising proposal laws...\n")
    find_proposal_law!(K, setup, solver, change_pt) #TODO remove this K, it's most likely not needed
    #TODO initialise for computation of gradients
end

"""
    check_if_recompute_ODEs(setup::DiffusionSetup)

Utility function for checking if H,Hν,c need to be re-computed for a respective
parameter update
"""
function check_if_recompute_ODEs(Ps, updt_coord)
    any([any([uc in depends_on_params(P) for uc in updt_coord]) for P in Ps])
end


#NOTE code to adapt:
#θ = params(P[1].Target)
#ϑs = [[θ[j] for j in idx(uc)] for uc in updtCoord]
#result = [DiffResults.GradientResult(ϑ) for ϑ in ϑs]
#resultᵒ = [DiffResults.GradientResult(ϑ) for ϑ in ϑs]
#Q = eltype(result)
