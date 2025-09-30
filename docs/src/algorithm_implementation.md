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

The **Strang splitting** scheme provides second-order accuracy by symmetrically arranging the substeps.

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