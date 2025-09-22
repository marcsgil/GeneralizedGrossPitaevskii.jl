# General Overview

This page provides a comprehensive overview of the mathematical framework and user interface conventions of GeneralizedGrossPitaevskii.jl. Whether you're simulating Bose-Einstein condensates, exciton-polariton systems, or exploring quantum phase-space methods, understanding these fundamentals will help you effectively use the package for your research.

## The Generalized Gross-Pitaevskii Equation

### Mathematical Form

The GeneralizedGrossPitaevskii.jl package solves equations of the form:

```math
i \, du = \left[ D(-i\nabla)u + V(\mathbf{r})u + G(u)u + i F(\mathbf{r}, t) \right] dt + \eta_1(u, \mathbf{r}) dW_1 + \eta_2(u, -i\nabla) dW_2
```

where:
- ``u(\mathbf{r}, t)`` is the complex (vector of) field(s)
- ``D(-i\nabla)`` represents the dispersion relation in momentum space
- ``V(\mathbf{r})`` is the external potential in position space
- ``G(u)`` captures nonlinear interactions depending on the field amplitude
- ``F(\mathbf{r}, t)`` represents time-dependent pumping or driving terms
- ``\eta_1, \eta_2`` are noise amplitude functions with Wiener increments ``dW_1, dW_2``

For multi-component systems, ``u`` becomes a vector of fields ``u = (u_1, u_2, \ldots, u_N)``, and the functions ``D``, ``V``, ``G`` can return matrices to describe coupling between components.

### Physical Interpretation

Each term in the generalized equation represents a distinct physical mechanism:

- **Dispersion term** ``D(-i\nabla)u``: Governs the kinetic energy and momentum-dependent dynamics. For non-relativistic particles, this is typically ``D(k) = \hbar k^2/(2m)``, leading to the familiar ``-\hbar^2\nabla^2/(2m)`` kinetic energy term.

  **Function signature:** `dispersion(k_tuple, param)` → scalar, vector or matrix
  - `k_tuple`: Tuple of momentum components `(kₓ, kᵧ, kᵤ, ...)`
  - Returns: Scalar for single-component systems, matrix for multi-component coupling
  - Example: `dispersion(ks, p) = sum(abs2, ks) / (2 * p.mass)`

- **Potential term** ``V(\mathbf{r})u``: External fields, confinement potentials, or spatially varying energy landscapes that influence the dynamics.

  **Function signature:** `potential(r_tuple, param)` → scalar, vector or matrix
  - `r_tuple`: Tuple of position components `(x, y, z, ...)`
  - Returns: Scalar for single-component systems, matrix for multi-component coupling
  - Example: `potential(rs, p) = 0.5 * p.ω² * sum(abs2, rs)`

- **Nonlinearity term** ``G(u)u``: Captures interactions between particles or field components. Common examples include the cubic nonlinearity ``G(u) = g|u|^2`` for contact interactions in Bose-Einstein condensates.

  **Function signature:** `nonlinearity(u_tuple, param)` → scalar, vector or matrix
  - `u_tuple`: Tuple of field components `(u₁, u₂, ...)`
  - Returns: Scalar for single-component, vector for multi-component systems
  - Example: `nonlinearity(u, p) = p.g * abs2(u[1])` (single-component cubic)

- **Pump/Drive term** ``F(\mathbf{r}, t)``: External pumping, driving, or dissipation. The factor of ``i`` allows both coherent driving (real ``F``) and incoherent gain/loss (imaginary ``F``).

  **Function signature:** `pump(r_tuple, param, t)` → scalar or vector
  - `r_tuple`: Tuple of position components `(x, y, z, ...)`
  - `param`: Parameter container
  - `t`: Current time
  - Returns: Scalar for single-component, vector for multi-component systems
  - Example: `pump(rs, p, t) = p.P₀ * exp(-sum(abs2, rs)/p.σ²)` (Gaussian pump)

- **Stochastic terms** ``\eta_j dW_j``: Quantum or thermal fluctuations that can be added in position space (``j=1``) or momentum space (``j=2``). These enable quantum phase-space methods like the truncated Wigner approximation.

  **Function signatures:**
  - Position noise: `position_noise_func(u_tuple, r_tuple, param)` → scalar, vector or matrix
  - Momentum noise: `momentum_noise_func(u_tuple, k_tuple, param)` → scalar, vector or matrix
  - Both functions take field components and spatial coordinates, returning noise amplitudes
  - Example: `position_noise_func(u, rs, p) = sqrt(p.γ)` (constant amplitude noise)

## Parameter Management

Parameters are passed to all user-defined functions via the `param` argument. This provides a flexible way to pass physical constants, coupling strengths, and other system-specific values.

**Recommended approach:**
- Use named tuples for organized parameter storage: `param = (; g=1.0, ω=2.0, γ=0.1)`
- Access parameters in functions: `potential(rs, p) = 0.5 * p.ω^2 * sum(abs2, rs)`
- Parameters can be `nothing` if not needed: `GrossPitaevskiiProblem(u0, lengths; dispersion)`

**Multi-component systems:**
- Use `@SMatrix` and `@SVector` from StaticArrays for efficient small matrices and vectors
- Example coupling matrix: `@SMatrix [g₁₁ g₁₂; g₁₂ g₂₂]`

**Identity functions:**
- `additiveIdentity` are used for zero terms (default for optional functions)
- These avoid unnecessary computations when terms are not needed