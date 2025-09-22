```@meta
EditURL = "../../examples/exciton_polariton.jl"
```

# Exciton-polariton system

In this example, we describe the polariton system as a two-component system consisting of excitons and photons inside a cavity.
This will teach us how to model and simulate the dynamics of systems with two or more coupled components.
The equations we want to solve are the following:

```math
i \frac{\partial}{\partial t} \begin{pmatrix} \psi_c \\ \psi_x \end{pmatrix} =
\begin{pmatrix} -\delta_c - i \gamma_c - \frac{\hbar^2 \nabla^2}{2m} & \Omega_r \\
\Omega_r & -\delta_x - i \gamma_x + g |\psi_x|^2 \end{pmatrix}
\begin{pmatrix} \psi_c \\ \psi_x \end{pmatrix}
+ \begin{pmatrix} F_c(x, t) \\ 0 \end{pmatrix}
```

In the above, `ψ_c` and `ψ_x` are the photonic and excitonic components of the polariton wavefunction, respectively,
`δ_c` and `δ_x` are the detunings of the cavity and exciton modes, `γ_c` and `γ_x` are their decay rates, `m` is the effective mass of the photons in the cavity,
`Ω_r` is the Rabi frequency describing the coupling between photons and excitons,
`F_c(x, t)` is the coherent pump acting on the photonic component, and `g` is the strength of the exciton-exciton nonlinear interactions.

As always, we start by loading the package

````@example exciton_polariton
using GeneralizedGrossPitaevskii
````

As we are modeling a two component system, there are some differences with respect to the single component case.
The main one is that, as the wavefunction now has two components,
the terms in the equation (dispersion, nonlinearity, and pump) must be defined as arrays.
This has to be done using the [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) package,
which allows us to define small fixed-size arrays efficiently, and, most importantly,
allows us to use them in GPU kernels.
For more information on why we use Static Arrays, see the [Multicomponent Systems](@ref) section of the documentation.

The StaticArrays.jl package is reexported by GeneralizedGrossPitaevskii.jl, so we can use it directly.
It's use is straightforward, usually only requiring the inclusion of the `@SVector` or `@SMatrix` macro in front of the array definition.

With that in mind, here is the dispersion term:

````@example exciton_polariton
function dispersion(k, param)
    Dcc = param.ħ * sum(abs2, k) / 2param.m - param.δc - im * param.γc
    Dxx = -param.δx - im * param.γx
    Dxc = param.Ωr
    @SMatrix [Dcc Dxc; Dxc Dxx]
end;
nothing #hide
````

We include the detuning and decay terms in the dispersion, as they are constants, as well as the coupling term `Ω_r` and the Laplacian in Fourier space.

In general, the nonlinear term is also a matrix, but in this case, it is diagonal.
Therefore, it is sufficient and more efficient to define only the diagonal elements as a vector.

````@example exciton_polariton
nonlinearity(ψ, param) = @SVector [0, param.g * abs2(ψ[2])];
nothing #hide
````

Finally, we define the pump profile, which is only applied to the photonic component.
We choose a Gaussian profile in space, but any other profile could be used.

````@example exciton_polariton
function pump(r, param, t)
    cpump = param.A * exp(-sum(abs2, r .- param.L / 2) / param.w^2)
    xpump = zero(cpump)
    @SVector [cpump, xpump]
end;
nothing #hide
````

Now, we define the parameters of the system and collect them in a named tuple.
Notice that, to specify a system with two spatial dimensions, we only need to make the lengths tuple have two elements, one for each dimension,
and the initial condition must be a tuple of two 2D arrays, one for each component.
The initial condition we choose here corresponds to an empty cavity.

````@example exciton_polariton
ħ = 0.654 # (meV*ps)
Ωr = 4
γx = 0.02
γc = 0.16
m = 1

δx = -2.56
δc = 0

A = 2
w = 100

g = 0.015

L = 256
N = 128
lengths = (L, L)

param = (; ħ, m, δc, γc, δx, γx, Ωr, A, w, g, L)
u0 = (zeros(ComplexF64, N, N), zeros(ComplexF64, N, N));
nothing #hide
````

Now, we define the problem, the number of saves, the time step, the time span, and the algorithm to use.
We then are ready to solve it.

````@example exciton_polariton
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity, pump, param)
nsaves = 64
dt = 5e-2
tspan = (0, 100)
alg = StrangSplitting()
ts, sol = solve(prob, alg, tspan; nsaves, dt, show_progress=false);
nothing #hide
````

The solution `sol` is a tuple of two arrays, one for each component.
Each array has dimensions `(N, N, nsaves)`, where the first two dimensions correspond to space and the last one to time.

To test our implementation, we can check that the steady-state solution obtained from the simulation matches the one obtained from the analytical expression.
For a homogeneous pump, the steady-state solution can be shown to obey the following relations:
```math
\left| \frac{(\delta_x + i \gamma_x - g n_x)(\delta_c + i \gamma_c)}{\Omega_r} - \Omega_r \right|^2 n_x = |F_c|^2 \\
\frac{|\delta_x + i \gamma_x - g n_x|^2}{\Omega_r^2}n_x = n_c
```

Although our pump is not homogeneous, it is approximately so in the center of the system, where the pump amplitude is close to `A`.
Therefore, we can check the above relations using the densities at the center of the system at the final time.

````@example exciton_polariton
nx = abs2.(last(sol))[N÷2, N÷2, end]
nc = abs2.(first(sol))[N÷2, N÷2, end];
nothing #hide
````

We can check that the relations are satisfied approximately.

This is the first relation:

````@example exciton_polariton
abs2(Ωr - (δx + im * γx - g * nx) * (δc + im * γc) / Ωr) * nx / abs2(A) # ≈ 1
````

This is the second relation:

````@example exciton_polariton
abs2(δx + im * γx - g * nx) * nx / Ωr^2 / nc # ≈ 1
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

