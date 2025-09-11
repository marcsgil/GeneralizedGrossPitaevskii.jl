using GeneralizedGrossPitaevskii, Documenter, Literate, DocumenterCitations

dir = pkgdir(GeneralizedGrossPitaevskii)
repo = "https://github.com/marcsgil/GeneralizedGrossPitaevskii.jl"

for file ∈ readdir(joinpath(dir, "examples"), join=true)
    if endswith(file, ".jl")
        Literate.markdown(file, joinpath(dir, "docs/src"); documenter=true, repo_root_url=joinpath(repo, "tree/master"))
    end
end

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(;
    modules=[GeneralizedGrossPitaevskii],
    authors="Marcos Gil",
    sitename="GeneralizedGrossPitaevskii.jl",
    pages=[
        "Home" => "index.md",
        "Quick-Start" => "quick_start.md",
        "Examples" => [
            "Damped Free Propagation" => "free_propagation_damping.md",
            "Bistability" => "bistability.md",
            "Exciton Polariton" => "exciton_polariton.md",
            "Truncated Wigner method" => "truncated_wigner.md",
        ],
        "Technical Documentation" => [
            "General Overview" => "general_overview.md",
            "Spatial Grid" => "spatial_grid.md",
            "Multicomponent Systems" => "multicomponent_systems.md",
            "Stochastic Simulations" => "stochastic_simulations.md",
            "Algorithm Implementation" => "algorithm_implementation.md",
            "GPU Support" => "gpu.md",
        ],
        "API" => "api.md",
        "References" => "references.md",],
    warnonly=true,
    plugins=[bib],
)

deploydocs(;
    repo="github.com/marcsgil/GeneralizedGrossPitaevskii.jl",
)