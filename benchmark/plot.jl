include(joinpath(@__DIR__, "bootstrap.jl"))
include(joinpath(@__DIR__, "reporting.jl"))

using CairoMakie

function plot_argument_value(prefix, default=nothing)
    matches = filter(arg -> startswith(arg, prefix), ARGS)
    isempty(matches) && return default
    length(matches) == 1 || error("$(prefix) may only be supplied once")
    split(only(matches), "="; limit=2)[2]
end

report_label(path) = splitext(basename(path))[1]

function shared_benchmark_ids(reports)
    indexes = report_index.(reports)
    ids = reduce(intersect, keys.(indexes))
    sort!(collect(ids)), indexes
end

function plot_individual_results(path; output)
    report = read_report(path)
    entries = sort!(collect(values(report_index(report))); by=entry -> entry["id"])
    isempty(entries) && error("the report has no successful benchmark cases")
    ids = getindex.(entries, "id")
    median_times_ms = [entry["trial"]["estimates"]["median"]["time_ns"] / 1e6 for entry in entries]
    figure_height = max(360, 120 + 36 * length(ids))
    fig = Figure(; size=(800, figure_height))
    ax = Axis(fig[1, 1];
        title="Median benchmark runtime: $(report_label(path))",
        xlabel="Median runtime (ms, log scale)", ylabel="Workload / backend / precision / measurement",
        xscale=log10, yticks=(eachindex(ids), ids),
    )
    scatter!(ax, median_times_ms, eachindex(ids); color=:steelblue, markersize=14)
    xlims!(ax, minimum(median_times_ms) / 2, maximum(median_times_ms) * 10)
    for (row, time_ms) in enumerate(median_times_ms)
        text!(ax, time_ms, row; text="$(round(time_ms; digits=3)) ms", align=(:left, :center),
            offset=(8, 0), color=:black,
        )
    end
    mkpath(dirname(output))
    save(output, fig)
    output
end

function plot_speedups(paths; output)
    length(paths) ≥ 2 || error("provide a baseline report followed by at least one candidate report")
    reports = read_report.(paths)
    baseline = first(reports)
    for candidate in Iterators.drop(reports, 1)
        compatible, reason = compatible_metadata(baseline, candidate)
        compatible || error("Refusing to plot incompatible benchmark reports: $(reason)")
    end

    ids, indexes = shared_benchmark_ids(reports)
    isempty(ids) && error("the reports have no successful benchmark cases in common")
    labels = report_label.(paths)
    speedups = Matrix{Float64}(undef, length(ids), length(reports))
    for (row, id) in enumerate(ids)
        baseline_time = indexes[1][id]["trial"]["estimates"]["median"]["time_ns"]
        for column in eachindex(reports)
            candidate_time = indexes[column][id]["trial"]["estimates"]["median"]["time_ns"]
            speedups[row, column] = baseline_time / candidate_time
        end
    end

    log_speedups = log2.(speedups)
    maximum_log_speedup = max(maximum(abs, log_speedups), 1.0)
    figure_height = max(360, 120 + 36 * length(ids))
    figure_width = max(640, 440 + 110 * length(reports))
    fig = Figure(; size=(figure_width, figure_height))
    ax = Axis(fig[1, 1];
        title="Median runtime speedup relative to $(first(labels))",
        xlabel="Benchmark report", ylabel="Workload / backend / precision / measurement",
        xticks=(eachindex(labels), labels), yticks=(eachindex(ids), ids),
        xticklabelrotation=pi / 8,
    )
    heatmap = heatmap!(ax, eachindex(labels), eachindex(ids), permutedims(log_speedups);
        colormap=:coolwarm, colorrange=(-maximum_log_speedup, maximum_log_speedup),
    )
    for row in eachindex(ids), column in eachindex(labels)
        speedup = speedups[row, column]
        text!(ax, column, row; text="$(round(speedup; digits=2))×", align=(:center, :center),
            color=abs(log2(speedup)) > maximum_log_speedup / 2 ? :white : :black,
        )
    end
    Colorbar(fig[1, 2], heatmap; label="log₂ speedup (higher is faster)")
    mkpath(dirname(output))
    save(output, fig)
    output
end

function main()
    report_paths = filter(arg -> !startswith(arg, "--"), ARGS)
    isempty(report_paths) && error("provide at least one benchmark report")
    default_filename = length(report_paths) == 1 ? "results.png" : "speedups.png"
    default_output = joinpath(@__DIR__, "results", default_filename)
    output = plot_argument_value("--output=", default_output)
    path = length(report_paths) == 1 ? plot_individual_results(only(report_paths); output) :
        plot_speedups(report_paths; output)
    println("Wrote plot to $(abspath(path))")
end

main()
