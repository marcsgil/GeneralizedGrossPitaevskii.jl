include(joinpath(@__DIR__, "bootstrap.jl"))
include(joinpath(@__DIR__, "workloads.jl"))

using Test

finite_solution(solution) = all(all(isfinite, Array(field)) for field in last(solution))

@testset "benchmark workloads" begin
    cpu_cases, skipped = build_cases(; backends=(:cpu,), precisions=(Float32, Float64))
    @test isempty(skipped)
    @test length(cpu_cases) == 6
    for case in cpu_cases
        @test finite_solution(run_solve(case))
        @test run_step(case) !== nothing
    end

    if cuda_available()
        for builder in (scalar_kerr_case, exciton_polariton_case)
            for T in (Float32, Float64)
                cpu_solution = run_solve(builder(T, :cpu))[2]
                gpu_solution = run_solve(builder(T, :cuda))[2]
                rtol = T === Float32 ? 5e-4 : 1e-10
                @test all(isapprox(gpu, cpu; rtol) for (gpu, cpu) in zip(Array.(gpu_solution), cpu_solution))
            end
        end
        stochastic = run_solve(truncated_wigner_case(Float32, :cuda))
        @test finite_solution(stochastic)
    end

    case = first(cpu_cases)
    report = Dict("schema_version" => 1, "metadata" => environment_metadata(),
        "benchmarks" => [report_entry(case, :solve, run(solve_benchmark(case); samples=1, seconds=0.01))])
    path = tempname() * ".json"
    write_report(report, path)
    parsed = read_report(path)
    @test parsed["schema_version"] == 1
    @test haskey(only(parsed["benchmarks"])["trial"], "estimates")
end
