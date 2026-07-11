using JSON

function write_report(report, path)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, report, 4)
    end
    path
end

function read_report(path)
    JSON.parsefile(path; dicttype=Dict{String,Any})
end

function report_index(report)
    Dict(entry["id"] => entry for entry in report["benchmarks"] if entry["status"] == "ok")
end

function compatible_metadata(before, after)
    before_meta, after_meta = before["metadata"], after["metadata"]
    for key in ("julia_version", "threads", "blas_threads", "cpu", "kernel")
        before_meta[key] == after_meta[key] || return false, "metadata differs for $(key)"
    end
    before_cuda, after_cuda = before_meta["cuda"], after_meta["cuda"]
    before_cuda["functional"] == after_cuda["functional"] || return false, "CUDA availability differs"
    if before_cuda["functional"]
        for key in ("name", "capability", "driver_version", "runtime_version")
            before_cuda[key] == after_cuda[key] || return false, "CUDA metadata differs for $(key)"
        end
    end
    true, nothing
end

function compare_reports(before_path, after_path)
    before, after = read_report(before_path), read_report(after_path)
    compatible, reason = compatible_metadata(before, after)
    compatible || error("Refusing to compare incompatible benchmark reports: $(reason)")

    before_index, after_index = report_index(before), report_index(after)
    shared = sort!(collect(intersect(keys(before_index), keys(after_index))))
    rows = Dict{String,Any}[]
    for id in shared
        baseline = before_index[id]["trial"]["estimates"]["median"]
        candidate = after_index[id]["trial"]["estimates"]["median"]
        push!(rows, Dict(
            "id" => id,
            "baseline_median_ns" => baseline["time_ns"],
            "candidate_median_ns" => candidate["time_ns"],
            "time_ratio" => candidate["time_ns"] / baseline["time_ns"],
            "speedup" => baseline["time_ns"] / candidate["time_ns"],
            "allocation_delta" => candidate["allocations"] - baseline["allocations"],
            "memory_delta_bytes" => candidate["memory_bytes"] - baseline["memory_bytes"],
        ))
    end
    Dict(
        "before" => abspath(before_path), "after" => abspath(after_path),
        "matched" => rows,
        "only_before" => sort!(collect(setdiff(keys(before_index), keys(after_index)))),
        "only_after" => sort!(collect(setdiff(keys(after_index), keys(before_index)))),
    )
end
