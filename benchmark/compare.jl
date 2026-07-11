include(joinpath(@__DIR__, "bootstrap.jl"))
include(joinpath(@__DIR__, "reporting.jl"))

length(ARGS) == 2 || error("usage: julia --project=benchmark benchmark/compare.jl BASELINE.json CANDIDATE.json")

comparison = compare_reports(ARGS[1], ARGS[2])
for row in comparison["matched"]
    println("$(row["id"]): $(round(row["speedup"]; digits=3))x speedup, ",
        "$(row["allocation_delta"]) allocation delta")
end
isempty(comparison["only_before"]) || println("Only in baseline: $(join(comparison["only_before"], ", "))")
isempty(comparison["only_after"]) || println("Only in candidate: $(join(comparison["only_after"], ", "))")
