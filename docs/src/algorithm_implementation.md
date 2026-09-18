# Algorithm Implementation

## Rationale for Split-Step Methods

The generalized Gross-Pitaevskii equation contains terms that are naturally handled in different representations:

- **Dispersion term** ``D(-i\nabla)u``: Most efficiently computed in momentum space using FFTs
- **Potential and nonlinearity terms** ``V(\mathbf{r})u + G(u)u``: Naturally computed in position space
- **Pump and noise terms**: Applied in position space

A direct numerical solution would require expensive spatial derivatives for the dispersion term and complex implicit methods for the nonlinearity. Split-step methods solve this by:

1. **Operator splitting**: Decompose the evolution into separate steps, each handled in its optimal representation
2. **Analytical solutions**: Each substep can often be solved exactly (dispersion) or very efficiently (local terms)
3. **Computational efficiency**: FFT-based dispersion steps are highly optimized and GPU-friendly
4. **Flexibility**: Easy to add new terms without restructuring the entire algorithm

The **Strang splitting** scheme symmetrically arranges the substeps. Second-order accuracy for deterministic evolution also requires sufficiently accurate substeps; the current implementation has the limitations described below.

## Accuracy limitations and future work

Within each position-space substep of duration ``h = dt/2``, the implementation applies ``e^{-ihG}e^{-ihV}`` in the same order. For noncommuting matrix-valued ``G`` and ``V``, this inner splitting is not symmetric and can reduce the overall deterministic method to first-order accuracy, even when both matrices are constant.

A future improvement is an inner Strang splitting, ``e^{-ihV/2}e^{-ihG}e^{-ihV/2}``: potential evolution for ``dt/4``, nonlinear evolution for ``dt/2``, then potential evolution for ``dt/4``, in both position-space substeps. This symmetrization is not yet implemented.

For field-dependent ``G(u)``, symmetrizing the factors alone does not guarantee second-order accuracy: the nonlinear evolution must itself be exact or approximated to at least second order. Freezing an arbitrary ``G(u)`` during a substep need not satisfy that requirement. Real scalar Kerr nonlinearity has an exact exponential nonlinear flow because that flow preserves the intensity; together with a real scalar potential and no pump or noise, it avoids this particular inner-splitting limitation.

These accuracy statements concern deterministic evolution, not the convergence order of the stochastic terms.

## Mathematical Formulation

The time evolution is implemented using a split-step method with Strang splitting. A single step of size `dt` from time `t` to `t+dt` is given by

```math
    u = e^{-iGdt/2} e^{-iVdt/2}(u + F(t)dt/2) + F(t+dt/2)dt/2 -i \sqrt{dt/2} \eta dW \\
    \tilde{u} = e^{-iDdt}\tilde{u}  \\
    u = e^{-iGdt/2} e^{-iVdt/2}(u + F(t+dt/2)dt/2) + F(t+dt)dt/2 -i \sqrt{dt/2} \eta dW \\
```

In the above, `u` denotes the fields in real space, while `ũ` denotes the fields in Fourier space; `G` is the nonlinear term, `V` is the potential term, and `D` is the dispersion term. The term `η` is the noise amplitude, and `dW` is a Wiener increment, which is a normally distributed random variable with mean `0` and variance `1`.

We can see that it is divided into three parts:

1. An evolution of `u` from time `t` to `t+dt / 2` performed in real space, without the dispersion term.

2. An evolution of `ũ` from time `t` to `t+dt`, performed in Fourier space, without the nonlinear, potential and pump terms.

3. An evolution of `u` from time `t+dt / 2` to `t+dt`, again performed in real space, without the dispersion term.
