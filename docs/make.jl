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
        ],
        "Theory" => [],
        "How-to Guides" => [],
        "API" => "api.md",
        "References" => "references.md",],
    warnonly=true,
    plugins=[bib],
)

deploydocs(;
    repo="github.com/marcsgil/GeneralizedGrossPitaevskii.jl",
)