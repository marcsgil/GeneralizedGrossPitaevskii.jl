# Benchmarking GeneralizedGrossPitaevskii.jl

This directory benchmarks the current `StrangSplitting` implementation before a Reactant.jl rewrite. It measures both a complete solve and one initialized solver step for scalar Kerr, coupled exciton-polariton, and stochastic truncated-Wigner workloads in `Float32` and `Float64`.

Run the suite from the repository root:

```sh
julia --threads=1 --project=benchmark benchmark/run.jl --output=benchmark/results/current.json
```

The benchmark environment targets Julia 1.12 and includes CUDA.jl. The default suite requests CPU and CUDA cases. Machines without a usable GPU still install CUDA.jl; their CUDA cases are automatically recorded as skipped when `CUDA.functional()` is false. On GPU hosts, kernel and FFT work is synchronized before timing is recorded.

Useful runner options:

```sh
# Fast smoke-sized run on CPU only
julia --threads=1 --project=benchmark benchmark/run.jl --quick --backend=cpu

# Set BenchmarkTools limits explicitly
julia --threads=1 --project=benchmark benchmark/run.jl --backend=cuda --seconds=10 --samples=100
```

Run correctness and JSON smoke checks with:

```sh
julia --threads=1 --project=benchmark benchmark/smoke.jl
```

To compare the baseline branch against a Reactant implementation, collect both reports on the same machine with identical backend, Julia thread count, and precision selection, then run:

```sh
julia --project=benchmark benchmark/compare.jl benchmark/results/current.json benchmark/results/reactant.json
```

The comparison refuses reports with mismatched host, CUDA device/driver, Julia version, or thread settings. Result JSON includes raw BenchmarkTools samples, median/minimum estimates, package versions, git revision/dirty state, and hardware metadata. Generated files in `benchmark/results/` are intentionally untracked.

Create a CairoMakie heatmap for two or more compatible reports. The first report is the baseline; each cell shows its median-runtime speedup relative to that report:

```sh
julia --project=benchmark benchmark/plot.jl --output=benchmark/results/speedups.png \
  benchmark/results/current.json benchmark/results/reactant.json
```

Pass one report instead to plot each successful case's absolute median runtime on a logarithmic millisecond axis:

```sh
julia --project=benchmark benchmark/plot.jl --output=benchmark/results/current.png \
  benchmark/results/current.json
```

The speedup comparison is steady-state: BenchmarkTools warms methods before timing. A future Reactant implementation should use these same workload identifiers and record first-call compilation time separately instead of mixing it into runtime measurements.
