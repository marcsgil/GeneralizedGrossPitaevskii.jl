include(joinpath(@__DIR__, "bootstrap.jl"))
include(joinpath(@__DIR__, "workloads.jl"))

using Dates

function argument_value(prefix, default=nothing)
    matches = filter(arg -> startswith(arg, prefix), ARGS)
    isempty(matches) && return default
    length(matches) == 1 || error("$(prefix) may only be supplied once")
    argument = only(matches)
    split(argument, "="; limit=2)[2]
end

function selected_backends(value)
    value == "cpu" && return (:cpu,)
    value == "cuda" && return (:cuda,)
    value == "all" && return (:cpu, :cuda)
    error("--backend must be cpu, cuda, or all")
end

function main()
    quick = "--quick" in ARGS
    seconds = parse(Float64, argument_value("--seconds=", quick ? "0.25" : "5.0"))
    samples = parse(Int, argument_value("--samples=", quick ? "3" : "100"))
    backends = selected_backends(argument_value("--backend=", "all"))
    default_output = joinpath(@__DIR__, "results", "benchmark-$(Dates.format(now(), "yyyymmdd-HHMMSS")).json")
    output = argument_value("--output=", default_output)

    report = run_benchmarks(; seconds, samples, backends)
    write_report(report, output)
    println("Wrote benchmark report to $(abspath(output))")
    for entry in report["benchmarks"]
        entry["status"] == "skipped" && println("Skipped $(entry["id"]): $(entry["reason"])")
        entry["status"] == "ok" || continue
        median_ns = entry["trial"]["estimates"]["median"]["time_ns"]
        println("$(entry["id"]): median $(round(median_ns / 1e6; digits=3)) ms")
    end
end

main()
