using BenchmarkTools
using CUDA
using GeneralizedGrossPitaevskii
using LinearAlgebra
using Pkg
using Random

import CommonSolve: init, solve, step!

include(joinpath(@__DIR__, "reporting.jl"))

const BENCHMARK_STEPS = 16

"""A fully specified solver workload used by both the runner and smoke tests."""
struct BenchmarkCase{P,T}
    id::String
    workload::Symbol
    backend::Symbol
    precision::DataType
    prob::P
    tspan::Tuple{T,T}
    dt::T
end

backend_array(u0, ::Val{:cpu}) = u0
backend_array(u0, ::Val{:cuda}) = CUDA.CuArray.(u0)

function scalar_kerr_case(::Type{T}, backend::Symbol) where {T<:AbstractFloat}
    N = 256
    L = T(32)
    dt = T(0.002)
    xs = range(zero(T), step=L / N, length=N)
    u0 = (Complex{T}.(exp.(-((x - L / 2)^2 + (y - L / 2)^2)) for x in xs, y in xs),)

    dispersion(k, p) = sum(abs2, k) / T(2)
    nonlinearity(ψ, p) = p.g * abs2(ψ[1])
    prob = GrossPitaevskiiProblem(
        backend_array(u0, Val(backend)), (L, L);
        dispersion, nonlinearity, param=(; g=T(-1)),
    )

    BenchmarkCase("scalar_kerr_2d_256", :scalar_kerr, backend, T, prob,
        (zero(T), BENCHMARK_STEPS * dt), dt)
end

function exciton_polariton_case(::Type{T}, backend::Symbol) where {T<:AbstractFloat}
    N = 256
    L = T(256)
    dt = T(0.05)
    u0 = (zeros(Complex{T}, N, N), zeros(Complex{T}, N, N))
    param = (; ħ=T(0.654), m=one(T), δc=zero(T), δx=T(-2.56),
        γc=T(0.16), γx=T(0.02), Ωr=T(4), A=T(2), w=T(100), g=T(0.015), L)

    function dispersion(k, p)
        Dcc = p.ħ * sum(abs2, k) / (T(2) * p.m) - p.δc - im * p.γc
        Dxx = -p.δx - im * p.γx
        @SMatrix [Dcc p.Ωr; p.Ωr Dxx]
    end
    nonlinearity(ψ, p) = @SVector [zero(T), p.g * abs2(ψ[2])]
    function pump(r, p, t)
        cpump = p.A * exp(-sum(abs2, r .- p.L / T(2)) / p.w^2)
        @SVector [cpump, zero(T)]
    end
    prob = GrossPitaevskiiProblem(
        backend_array(u0, Val(backend)), (L, L); dispersion, nonlinearity, pump, param,
    )

    BenchmarkCase("exciton_polariton_2d_256", :exciton_polariton, backend, T, prob,
        (zero(T), BENCHMARK_STEPS * dt), dt)
end

function truncated_wigner_case(::Type{T}, backend::Symbol) where {T<:AbstractFloat}
    N = 1024
    trajectories = 128
    L = T(512)
    dt = T(0.01)
    dx = L / N
    rng = MersenneTwister(20260711)
    u0 = (randn(rng, Complex{T}, N, trajectories) / sqrt(T(2) * dx),)
    noise_prototype = similar.(u0)
    param = (; ħ=T(0.6582), γ=T(0.047 / 0.6582), m=T(1 / 6),
        g=T(3e-4 / 0.6582), δ=T(0.49 / 0.6582), A=T(10), dx)

    dispersion(k, p) = p.ħ * sum(abs2, k) / (T(2) * p.m) - p.δ - im * p.γ / T(2)
    nonlinearity(ψ, p) = p.g * (abs2(ψ[1]) - inv(p.dx))
    pump(r, p, t) = p.A
    position_noise_func(ψ, r, p) = sqrt(p.γ / (T(2) * p.dx))
    prob = GrossPitaevskiiProblem(
        backend_array(u0, Val(backend)), (L,);
        dispersion, nonlinearity, pump, position_noise_func,
        noise_prototype=backend_array(noise_prototype, Val(backend)), param,
    )

    BenchmarkCase("truncated_wigner_1d_1024x128", :truncated_wigner, backend, T, prob,
        (zero(T), BENCHMARK_STEPS * dt), dt)
end

const CASE_BUILDERS = (scalar_kerr_case, exciton_polariton_case, truncated_wigner_case)

function cuda_available()
    try
        CUDA.functional()
    catch
        false
    end
end

function cuda_unavailable_reason()
    cuda_available() && return nothing
    "CUDA.functional() returned false"
end

function build_cases(; backends=(:cpu, :cuda), precisions=(Float32, Float64))
    cases = BenchmarkCase[]
    skipped = Dict{String,Any}[]
    for backend in backends
        if backend === :cuda && !cuda_available()
            for T in precisions, builder in CASE_BUILDERS
                case_id = builder === scalar_kerr_case ? "scalar_kerr_2d_256" :
                    builder === exciton_polariton_case ? "exciton_polariton_2d_256" :
                    "truncated_wigner_1d_1024x128"
                push!(skipped, Dict(
                    "id" => "$(case_id)/cuda/$(T)", "status" => "skipped",
                    "reason" => "CUDA is not functional on this host",
                ))
            end
            continue
        end
        for T in precisions, builder in CASE_BUILDERS
            push!(cases, builder(T, backend))
        end
    end
    cases, skipped
end

solve_kwargs(case) = (; dt=case.dt, nsaves=1, save_start=false, show_progress=false)

function synchronize_backend(backend::Symbol)
    backend === :cuda && CUDA.synchronize()
    nothing
end

function run_solve(case::BenchmarkCase)
    result = solve(case.prob, StrangSplitting(), case.tspan; solve_kwargs(case)...)
    synchronize_backend(case.backend)
    result
end

function run_step(case::BenchmarkCase)
    iterator = init(case.prob, StrangSplitting(), case.tspan; solve_kwargs(case)...)
    step!(iterator, case.dt, case.dt)
    synchronize_backend(case.backend)
    iterator
end

function solve_benchmark(case::BenchmarkCase)
    prob, tspan, kwargs = case.prob, case.tspan, solve_kwargs(case)
    if case.backend === :cuda
        return @benchmarkable begin
            result = solve($prob, StrangSplitting(), $tspan; $kwargs...)
            synchronize_backend(:cuda)
            result
        end evals=1
    end
    @benchmarkable solve($prob, StrangSplitting(), $tspan; $kwargs...) evals=1
end

function step_benchmark(case::BenchmarkCase)
    prob, tspan, kwargs, dt = case.prob, case.tspan, solve_kwargs(case), case.dt
    if case.backend === :cuda
        return @benchmarkable begin
            step!(_iterator, $dt, $dt)
            synchronize_backend(:cuda)
        end setup=(_iterator = init($prob, StrangSplitting(), $tspan; $kwargs...)) evals=1
    end
    @benchmarkable step!(_iterator, $dt, $dt) setup=(_iterator = init($prob, StrangSplitting(), $tspan; $kwargs...)) evals=1
end

trial_estimate_dict(estimate) = Dict(
    "time_ns" => estimate.time,
    "gctime_ns" => estimate.gctime,
    "memory_bytes" => estimate.memory,
    "allocations" => estimate.allocs,
)

function trial_dict(trial)
    Dict(
        "times_ns" => trial.times,
        "gctimes_ns" => trial.gctimes,
        "memory_bytes" => trial.memory,
        "allocations" => trial.allocs,
        "estimates" => Dict(
            "minimum" => trial_estimate_dict(minimum(trial)),
            "median" => trial_estimate_dict(median(trial)),
        ),
    )
end

function package_version(mod)
    mod === nothing && return "unavailable"
    version = Base.pkgversion(mod)
    version === nothing ? "unknown" : string(version)
end

function git_value(args...)
    try
        command = Cmd(["git", args...])
        readchomp(Cmd(command; dir=normpath(joinpath(@__DIR__, ".."))))
    catch
        "unknown"
    end
end

function cuda_metadata()
    cuda_available() || return Dict("functional" => false, "reason" => cuda_unavailable_reason())
    device = CUDA.device()
    Dict(
        "functional" => true,
        "name" => CUDA.name(device),
        "capability" => string(CUDA.capability(device)),
        "total_memory_bytes" => CUDA.totalmem(device),
        "driver_version" => string(CUDA.driver_version()),
        "runtime_version" => string(CUDA.runtime_version()),
    )
end

function environment_metadata()
    Dict(
        "julia_version" => string(VERSION),
        "package_versions" => Dict(
            "GeneralizedGrossPitaevskii" => package_version(GeneralizedGrossPitaevskii),
            "BenchmarkTools" => package_version(BenchmarkTools),
            "CUDA" => package_version(CUDA),
        ),
        "threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "cpu" => Sys.CPU_NAME,
        "kernel" => Sys.KERNEL,
        "git_revision" => git_value("rev-parse", "HEAD"),
        "git_dirty" => !isempty(git_value("status", "--porcelain")),
        "cuda" => cuda_metadata(),
    )
end

case_key(case::BenchmarkCase, measurement::Symbol) =
    "$(case.id)/$(case.backend)/$(case.precision)/$(measurement)"

function report_entry(case::BenchmarkCase, measurement::Symbol, trial)
    Dict(
        "id" => case_key(case, measurement),
        "status" => "ok",
        "workload" => string(case.workload),
        "backend" => string(case.backend),
        "precision" => string(case.precision),
        "measurement" => string(measurement),
        "steps" => BENCHMARK_STEPS,
        "trial" => trial_dict(trial),
    )
end

function run_benchmarks(; seconds=5.0, samples=100, backends=(:cpu, :cuda), precisions=(Float32, Float64))
    cases, skipped = build_cases(; backends, precisions)
    entries = Dict{String,Any}[skipped...]
    for case in cases
        for (measurement, benchmark) in ((:solve, solve_benchmark(case)), (:step, step_benchmark(case)))
            trial = run(benchmark; seconds, samples)
            push!(entries, report_entry(case, measurement, trial))
        end
    end
    Dict("schema_version" => 1, "metadata" => environment_metadata(), "benchmarks" => entries)
end
