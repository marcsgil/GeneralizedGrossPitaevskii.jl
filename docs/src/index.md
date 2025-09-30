# GeneralizedGrossPitaevskii.jl

GeneralizedGrossPitaevskii.jl is a Julia package for simulating a wide class of quantum fluid dynamics, from Bose-Einstein condensates to exciton-polariton systems, driven-dissipative fluids, and beyond.

## Why GeneralizedGrossPitaevskii.jl?

This package unifies the simulation of diverse quantum fluid systems under a single, high-performance framework. With this package you can:

- **Focus on physics, not numerics**: Describe complex systems with simple function definitions
- **Include quantum effects**: Built-in support for phase space methods that allow the inclusion of quantum fluctuations through stochastic terms (e.g., Truncated Wigner approximation, Positive-P representation)
- **Handle multi-component systems**: Native support for coupled fields with elegant StaticArrays integration
- **Scale effortlessly**: Same code runs efficiently on both CPU and GPU via KernelAbstractions.jl
- **Work with one consistent interface**: No need to learn different packages for different physical systems

The package solves a generalized Gross-Pitaevskii equation:

```math
i d u(r, t) = [D(-i\nabla)u + V(r)u + G(u)u + i F(r, t)]dt + \eta(u, r) dW
```

This framework encompasses a vast range of quantum fluid phenomena, allowing researchers to concentrate on physics rather than numerical implementation details.

## Getting Started

Check out the [Quick-Start](@ref) guide to begin using the package, or explore the Examples (linked in the menu to the left) to see various physical systems in action. Their source files can be found in the [`examples/`](https://github.com/marcsgil/GeneralizedGrossPitaevskii.jl/tree/main/examples) directory of the GitHub repository.