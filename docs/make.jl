#push!(LOAD_PATH,"../src/")
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
            "Setup" => joinpath("man", "setup.md"),
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
        "Visualisation tools" => joinpath("vis", "visualisation.md")
    ],
)

deploydocs(
    repo = "github.com/mmider/BridgeSDEInference.jl.git",
)
