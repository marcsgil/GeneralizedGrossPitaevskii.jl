# GeneralizedGrossPitaevskii.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://marcsgil.github.io/GeneralizedGrossPitaevskii.jl/stable/)

A high-performance Julia package for simulating generalized Gross-Pitaevskii equations, designed for quantum fluid dynamics including Bose-Einstein condensates, exciton-polariton systems, and quantum phase-space methods.

## Why GeneralizedGrossPitaevskii.jl?

This package unifies the simulation of diverse quantum fluid systems under a single, high-performance framework:

- **🔬 Focus on physics, not numerics**: Describe complex systems with simple function definitions
- **🌊 Include quantum effects**: Built-in support for phase space methods that allow the inclusion of quantum fluctuations through stochastic terms (e.g., Truncated Wigner approximation, Positive-P representation)
- **🔗 Handle multi-component systems**: Native support for coupled fields with elegant StaticArrays integration
- **⚡ Scale effortlessly**: Same code runs efficiently on both CPU and GPU via KernelAbstractions.jl
- **🎯 Work with one consistent interface**: No need to learn different packages for different physical systems

## Mathematical Framework

The package solves equations of the form:

```
i du(r, t) = [D(-i∇)u + V(r)u + G(u)u + i F(r, t)]dt + η₁(u, r) dW₁ + η₂(u, -i∇) dW₂
```

where:
- `u(r, t)` is the complex field(s)
- `D(-i∇)` represents the dispersion relation in momentum space
- `V(r)` is the external potential in position space
- `G(u)` captures nonlinear interactions depending on the field amplitude
- `F(r, t)` represents time-dependent pumping or driving terms
- `η₁, η₂` are noise amplitude functions with Wiener increments `dW₁, dW₂`

This framework encompasses a vast range of quantum fluid phenomena.

## Quick Start

### Installation

```julia
using Pkg
Pkg.add("GeneralizedGrossPitaevskii")
```

### Basic Usage

```julia
using GeneralizedGrossPitaevskii

# Define 2D simulation grid
N = 128
L = 8.0
lengths = (L, L)

# Create initial Gaussian wavefunction
ΔL = L / N
rs = StepRangeLen(0, ΔL, N)
u0 = ([exp(-(x - L/2)^2 - (y - L/2)^2) for x in rs, y in rs],)

# Define dispersion relation (kinetic energy)
dispersion(ks, param) = sum(abs2, ks) / 2

# Create problem
prob = GrossPitaevskiiProblem(u0, lengths; dispersion)

# Solve with Strang splitting
ts, sol = solve(prob, StrangSplitting(), (0, 1.0); dt=0.01, nsaves=64)
```

### Adding Nonlinearity

```julia
# Add cubic nonlinearity: i ∂u/∂t = -∇²u/2 + g|u|²u
nonlinearity(u, param) = param.g * abs2(u[1])
param = (g = -6.0,)

prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity, param)
ts, sol = solve(prob, StrangSplitting(), (0, 0.4); dt=0.01, nsaves=64)
```

## Key Features

### Multi-Component Systems
```julia
# Two-component exciton-polariton system
function dispersion(k, p)
    Dcc = p.ħ * sum(abs2, k) / (2 * p.m) - p.δc - im * p.γc
    Dxx = -p.δx - im * p.γx
    @SMatrix [Dcc p.Ωr; p.Ωr Dxx]
end

nonlinearity(u, p) = @SVector [0, p.g * abs2(u[2])]
```

### Quantum Fluctuations
```julia
# Truncated Wigner method with position noise
position_noise_func(u, r, p) = sqrt(p.noise_strength)
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity,
                              position_noise_func, param)
```

### GPU Acceleration
```julia
using CUDA
u0_gpu = (CuArray(u0[1]),)
prob_gpu = GrossPitaevskiiProblem(u0_gpu, lengths; dispersion, nonlinearity, param)
# Simulation automatically runs on GPU
```

## Examples

The package includes comprehensive examples demonstrating various physical systems:

- **[Quick Start](examples/quick_start.jl)**: Basic 2D Gaussian evolution with/without nonlinearity
- **[Free Propagation with Damping](examples/free_propagation_damping.jl)**: Adding imaginary dispersion terms for dissipation
- **[Bistability](examples/bistability.jl)**: Polariton condensate bistability with time-dependent pumping
- **[Exciton-Polariton System](examples/exciton_polariton.jl)**: Two-component system with matrix dispersion relations
- **[Truncated Wigner Method](examples/truncated_wigner.jl)**: Comprehensive quantum simulation with stochastic trajectories

## Documentation

Complete documentation with detailed examples and API reference is available at:
**[https://marcsgil.github.io/GeneralizedGrossPitaevskii.jl/stable/](https://marcsgil.github.io/GeneralizedGrossPitaevskii.jl/stable/)**

## Performance

- **Type-stable design** with parametric structs for optimal performance
- **GPU acceleration** through KernelAbstractions.jl
- **Efficient memory management** with pre-allocated buffers
- **FFT-based spatial derivatives** for spectral accuracy
- **Strang splitting** algorithm for robust time integration

## Requirements

- Julia ≥ 1.6
- Compatible with CPU and GPU backends (CUDA)
- Periodic boundary conditions via FFT

## Contributing

Contributions are welcome!

## Citation

If you use this package in your research, please consider citing:

```bibtex
@software{Gil2024GeneralizedGrossPitaevskii,
  author = {Gil, Marcos},
  title = {GeneralizedGrossPitaevskii.jl: A Julia Package for Quantum Fluid Dynamics},
  url = {https://github.com/marcsgil/GeneralizedGrossPitaevskii.jl},
  version = {0.1.0},
  year = {2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This package builds upon the excellent Julia ecosystem, particularly:
- [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) for GPU abstraction
- [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) for fast Fourier transforms
- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) for efficient small arrays
- [CommonSolve.jl](https://github.com/SciML/CommonSolve.jl) for the solver interface