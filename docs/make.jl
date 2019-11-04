using Documenter, BridgeSDEInference

makedocs(
    modules = [BridgeSDEInference],
    format = Documenter.HTML(
        #prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    sitename="BridgeSDEInference.jl",
    pages = [
        "Home" => "index.md",
        "Tutorial" => Any[
            "Workflow" => joinpath("man", "overview.md"),
            "Defining a diffusion" => joinpath("man", "model_definition.md"),
            "Data generation" => joinpath("man", "generate_data.md"),
            "Diffusion Setup" => joinpath("man", "diffusion_setup.md"),
            "MCMC Setup" => joinpath("man", "mcmc_setup.md"),
            "Running the sampler" => joinpath("man", "run.md"),
            "Querying the results" => joinpath("man", "query.md"),
            ],
        "Examples" => Any[
            "FitzHugh-Nagumo model" => joinpath("examples", "fitzhugh_nagumo.md"),
            "Jansen-Rit model" => joinpath("examples", "jansen_rit.md"),
            "Lorenz63 system" => joinpath("examples", "lorenz63.md"),
            "Lorenz96 system" => joinpath("examples", "lorenz96.md"),
            "Prokaryotic autoregulatory gene network" => joinpath("examples", "prokaryote.md"),
            "Sine diffusion" => joinpath("examples", "sine.md")
        ],
        "Visualisation tools" => joinpath("vis", "visualisation.md"),
        "Extras" => Any[
            "Generic MCMC" => joinpath("extras", "generic_mcmc.md"),
            "First passage times" => joinpath("extras", "first_passage_times.md"),
            "Blocking" => joinpath("extras", "blocking.md")
            ],
        "Adaptations" => Any[
            "Adaptive proposals" => joinpath("adaptations", "adaptive_proposals.md"),
            "Fusion" => joinpath("adaptations", "fusion.md")
        ]
    ],
)

deploydocs(
    repo = "github.com/mmider/BridgeSDEInference.jl.git",
)
