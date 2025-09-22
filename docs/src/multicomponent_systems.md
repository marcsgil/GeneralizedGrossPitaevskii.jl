# Multicomponent Systems

Many quantum fluid systems involve multiple coupled fields, such as multi-component Bose-Einstein condensates or exciton-polariton systems. GeneralizedGrossPitaevskii.jl provides native support for these systems through efficient [StaticArrays.jl](https://juliaarrays.github.io/StaticArrays.jl/stable/) integration, enabling both CPU and GPU simulations.

This page explains how to set up and work with multicomponent systems, from field initialization to operator definitions.

## Specifying Multicomponent Fields

Multicomponent systems are defined by providing multiple field components as a tuple of arrays. Each component represents a different physical field.

### Field Initialization

```julia
# Two-component system (e.g., exciton-polariton)
N = 64
L = 10.0
ΔL = L / N
rs = StepRangeLen(0, ΔL, N)

# Component 1: Photonic field (initially empty)
ψ_c = zeros(ComplexF64, N, N)

# Component 2: Excitonic field (Gaussian profile)  
ψ_x = [exp(-(x - L/2)^2 - (y - L/2)^2) for x in rs, y in rs]

# Multicomponent initial condition
u0 = (ψ_c, ψ_x)
```

### Component Organization

- **Fields are ordered**: The order in the tuple `u0 = (ψ₁, ψ₂, ...)` determines component indexing
- **Consistent dimensions**: All field components must have the same spatial dimensions
- **Type consistency**: Components should have compatible element types (typically `ComplexF64`)

## Specifying Multicomponent Operators

For multicomponent systems, some of the terms that define the equation (dispersion, potential, nonlinearity) are matrices, while the pump is a vector. 

As an example, the dispersion term for a two-component exciton-polariton system can be defined as:

```julia
function dispersion(k, param)
    Dcc = param.ħ * sum(abs2, k) / 2param.m - param.δc - im * param.γc
    Dxx = -param.δx - im * param.γx
    Dxc = param.Ωr
    @SMatrix [Dcc Dxc; Dxc Dxx]
end
```

We see here that it is a 2x2 matrix, with the diagonal elements corresponding to the photonic and excitonic components, and the off-diagonal elements representing their coupling. One just needs to ensure the inclusion of the `@SMatrix` macro reexported from [StaticArrays.jl](https://juliaarrays.github.io/StaticArrays.jl/stable/) in front of the usual matrix definition.

From the same example, the nonlinearity is diagonal, so it is sufficient to define only the diagonal elements as a vector:

```julia
nonlinearity(ψ, param) = @SVector [0, param.g * abs2(ψ[2])]
```

This is a general feature of multicomponent systems in GeneralizedGrossPitaevskii.jl: diagonal operators can be defined as vectors, while full operators must be defined as matrices. Furthermore, if the operator is a multiple of the identity matrix, it can be simply defined as a scalar.