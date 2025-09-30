# Stochastic Simulations

Stochastic terms in the generalized Gross-Pitaevskii equation enable the simulation of quantum fluctuations and thermal effects through phase-space methods. GeneralizedGrossPitaevskii.jl supports noise terms in both position and momentum space, allowing for techniques such as the truncated Wigner approximation and Positive-P representation.

This page explains how to implement and configure stochastic terms for simulations.

## Noise Implementation

Stochastic terms are added to the deterministic Gross-Pitaevskii equation through noise amplitude functions that multiply Wiener processes. The general form is

```math
i \, du = [...] \, dt + \eta(u, \mathbf{r}) \, dW
```

To enable stochastic simulations, you must provide:

1. **Noise amplitude function**: `position_noise_func`
2. **Noise prototype**: Template arrays defining noise structure and type
3. **Modified problem constructor**: Include noise terms in `GrossPitaevskiiProblem`

### Basic Setup

```julia
# Define noise prototype (same structure as initial condition)
noise_prototype = similar.(u0)

# Position noise function
position_noise_func(u, r, param) = sqrt(param.γ / param.dx)

# Create problem with noise
prob = GrossPitaevskiiProblem(u0, lengths; 
    dispersion, nonlinearity, pump, param,
    noise_prototype, position_noise_func)
```

### Ensemble Simulations

For quantum phase-space methods, multiple stochastic trajectories are typically required. They can be implemented by adding an ensemble dimension to the initial condition:

```julia
# Add ensemble dimension (e.g., 100 trajectories)
u0_ensemble = (randn(ComplexF64, N, 100),)

# Noise affects each trajectory independently
prob = GrossPitaevskiiProblem(u0_ensemble, lengths; 
    dispersion, nonlinearity, pump, param,
    noise_prototype, position_noise_func)
```

### Position Space Noise

Position space noise is applied directly to the field in real space and is commonly used for:
- **Truncated Wigner method**: Quantum vacuum fluctuations
- **Thermal noise**: Temperature-dependent fluctuations  

**Function signature:**
```julia
position_noise_func(u_tuple, r_tuple, param) → scalar or vector
```

**Common implementations:**

**Constant noise amplitude:**
```julia
position_noise_func(u, r, param) = sqrt(param.γ / param.dx)
```

**Field-dependent noise:**
```julia
# Noise proportional to local field amplitude (multiplicative noise)
position_noise_func(u, r, param) = param.α * abs(u[1])
```

**Spatially varying noise:**
```julia
# Noise with spatial profile (e.g., for pumped regions)
position_noise_func(u, r, param) = param.β * exp(-sum(abs2, r) / param.σ²)
```

**Multicomponent noise:**
```julia
# Different noise for each component
function position_noise_func(u, r, param)
    @SVector [sqrt(param.γ₁ / param.dx), sqrt(param.γ₂ / param.dx)]
end
```

### Noise Prototypes

The `noise_prototype` parameter defines the structure, type, and size of the noise arrays used in stochastic integration. It must be a tuple with the same number of elements as the initial condition `u0`, and each element must have the same shape as the field components in `u0`, but may have a different type (e.g., real vs complex).

**Basic usage:**
```julia
# Single-component system
u0 = (zeros(ComplexF64, N),)
noise_prototype = similar.(u0)

# Multi-component system  
u0 = (zeros(ComplexF64, N), zeros(ComplexF64, N))
noise_prototype = similar.(u0)
```

**Ensemble simulations:**
```julia  
# Multiple stochastic trajectories (last dimension = ensemble)
u0 = (zeros(ComplexF64, N, n_trajectories),)
noise_prototype = similar.(u0)

# Each trajectory gets independent noise realizations
```

**Type considerations:**
```julia
# Complex noise 
noise_prototype = (zeros(ComplexF64, N),)

# Real noise
noise_prototype = (zeros(Float64, N),)

# GPU arrays
using CUDA
u0 = ((CUDA.zeros(ComplexF64, N)),)
noise_prototype = similar.(u0)  # Automatically CUDA arrays
```

### Important Notes

- **Memory allocation**: The prototype determines memory layout and GPU/CPU placement
- **Ensemble independence**: Each trajectory in the ensemble dimension receives independent noise
- **Size matching**: Prototype must have the same spatial dimensions as field components

### Performance Tips

- Use appropriate precision (Float32 vs Float64) for your hardware
- For GPU simulations, ensure prototype arrays are GPU-resident