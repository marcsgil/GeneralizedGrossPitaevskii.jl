# Spatial Grid

GeneralizedGrossPitaevskii.jl uses uniform Cartesian grids with periodic boundary conditions implemented via Fast Fourier Transforms (FFTs). This page explains the grid conventions, coordinate systems, and how to set up spatial domains for your simulations.

## Grid Conventions

### Physical Domain Setup

The spatial domain is defined by the `lengths` parameter, which is a tuple specifying the physical size in each spatial dimension:

```julia
# 1D domain of size L
lengths = (L,)

# 2D domain of size Lₓ × Lᵧ  
lengths = (Lₓ, Lᵧ)

# 3D domain of size Lₓ × Lᵧ × Lᵤ
lengths = (Lₓ, Lᵧ, Lᵤ)
```

### Grid Points and Spacing

For a domain of size `L` with `N` grid points, the package uses:

- **Grid spacing**: `ΔL = L / N`
- **Grid points**: `0, ΔL, 2ΔL, ..., (N-1)ΔL`
- **Grid creation**: `StepRangeLen(0, ΔL, N)`

The grid always starts at zero and extends to `L - ΔL`, which is consistent with periodic boundary conditions where the point at `L` is equivalent to the point at `0`.

### Position Space (Direct Grid)

For user convenience when setting up initial conditions or spatially varying functions, you can create position grids manually:

```julia
N = 64
L = 10.0
ΔL = L / N
rs = StepRangeLen(0, ΔL, N)

# Create initial condition
u0 = [exp(-(x - L/2)^2) for x in rs]  # 1D Gaussian
u0 = [exp(-(x - L/2)^2 - (y - L/2)^2) for x in rs, y in rs]  # 2D Gaussian
```

Internally, the package provides `direct_grid(prob)` which returns a tuple of ranges for each spatial dimension.

### Momentum Space (Reciprocal Grid)

For dispersion relations, the package automatically generates momentum grids using FFT frequency conventions:

- **1D momentum**: `k = fftfreq(N, 2π * N/L)`
- **Allowed momenta are discrete**: `k = 2πn/L` where `n` is integer
- **Momentum resolution**: `Δk = 2π/L`
- **Range**: `k ∈ [-π N/L, π N/L)` with proper FFT ordering
- **Nyquist frequency (Maximum resolved momentum)**: `kₘₐₓ = π N/L`

The reciprocal grid is accessed internally `reciprocal_grid(prob)` and follows standard FFT conventions where negative frequencies come after positive ones.

## Boundary Conditions

GeneralizedGrossPitaevskii.jl exclusively uses **periodic boundary conditions** implemented through Fast Fourier Transforms. This choice provides several advantages:

### Why Periodic Boundaries?

- **FFT Efficiency**: FFTs naturally assume periodicity
- **GPU Compatibility**: Highly optimized FFT libraries (FFTW, CUDA) work seamlessly
- **Simplicity**: Easier to implement and maintain for arbitrary dimensionality compared to other boundary conditions

### Practical Implications

**Domain Size Considerations:**
- Choose domain size `L` large enough that the wavefunction is negligible at boundaries or consider using loss terms to dampen boundary effects
- For localized states (e.g., solitons), ensure `L ≫ characteristic_length_scale`
- For periodic structures, align domain size with the natural periodicity