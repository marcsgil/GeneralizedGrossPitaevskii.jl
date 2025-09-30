```@meta
EditURL = "../../examples/quick_start.jl"
```

# Quick Start
In this quick start example, we will demonstrate how to set up a simple Gross-Pitaevskii problem
and solve it using the GeneralizedGrossPitaevskii package.
As a first example, we will simulate free propagation of a wavefunction in a 2D grid.
Later, we will add a nonlinear term in order to have a proper Gross-Pitaevskii equation.

## Free Propagation

We first load a few necessary packages.

````@example quick_start
using GeneralizedGrossPitaevskii, CairoMakie, StructuredLight
````

CairoMakie and StructuredLight are used for visualization.

The first step is to define the parameters of the grid of our simulation.
We will simulate a 2D system with a grid with N x N points with size of L x L.

````@example quick_start
N = 128
L = 8;
nothing #hide
````

Now we define a lengths parameter:

````@example quick_start
lengths = (L, L);
nothing #hide
````

This is a tuple that defines the physical dimensions of the simulation domain.
Each element corresponds to the size of the grid in each dimension.
As we will be simulating a 2D system, and the grid is square, both elements are equal.

Now, we turn to the definition of the initial condition of the wavefunction.

The first step is to define the inter-point distance in the grid:

````@example quick_start
ΔL = L / N;
nothing #hide
````

This is used to create a range of points in the grid.

````@example quick_start
rs = StepRangeLen(0, ΔL, N);
nothing #hide
````

Observe that this package always assumes that the grid is of the form 0, ΔL, 2ΔL, ..., (N-1)ΔL.
This is easily created with the `StepRangeLen` function.

We can use this grid to create an initial condition for the wavefunction:

````@example quick_start
initial_condition = [exp(-(x - L / 2)^2 - (y - L / 2)^2) for x in rs, y in rs];
nothing #hide
````

This creates a 2D array representing a Gaussian wavefunction centered at the middle of the grid.

Let us visualize the initial condition:

````@example quick_start
visualize(abs2.(initial_condition))
````

As this package is able to handle systems with multiple wavefunctions, we need to wrap the initial condition in a tuple with one element:

````@example quick_start
u0 = (initial_condition,);
nothing #hide
````

This indicates that we have one wavefunction in our simulation.

Next, we need to specify the equation that our system obeys.
As this is a free propagation problem, the wavefunction satisfies the Schrödinger equation i ∂u(r, t)/∂t = -∇²u / 2.
In Fourier space, this corresponds to a dispersion relation D(k) = |k|² / 2, which we now define:

````@example quick_start
dispersion(ks, param) = sum(abs2, ks) / 2;
nothing #hide
````

The extra argument `param` is a placeholder for additional parameters, such as mass, that may be needed in more complex problems,
but is not used here.

Now we can create the problem instance:

````@example quick_start
prob = GrossPitaevskiiProblem(u0, lengths; dispersion);
nothing #hide
````

This encapsulates all the necessary components to define the Gross-Pitaevskii equation.

Now that we have defined the problem, we can solve it.
We just need to specify some additional parameters:
This is the algorithm we will use to solve the problem:

````@example quick_start
alg = StrangSplitting();
nothing #hide
````

Right now, this is the only algorithm implemented in this package, so there isn't much choice.

This algorithm uses a fixed time step, which we can define:

````@example quick_start
dt = 0.01;
nothing #hide
````

Next, we define the time span of the simulation:

````@example quick_start
tspan = (0, 1);
nothing #hide
````

This indicates that we want to simulate the system from time t=0 to t=1

Now, we indicate how many times we want to save the solution during the simulation:

````@example quick_start
nsaves = 64;
nothing #hide
````

Finally, we can solve the problem:

````@example quick_start
ts, sol = solve(prob, alg, tspan; nsaves, dt);
nothing #hide
````

The `solve` function returns two values: the time points `ts` on which the solution was saved and the solution `sol`.
`sol` is again a tuple, where each element corresponds to a wavefunction in the simulation.
Each element in this tuple is an array with one extra dimension than the initial condition, which corresponds to the time dimension.
We can use the `save_animation` function (which comes from StructuredLight and CairoMakie) to visualize the solution:

````@example quick_start
save_animation(abs2.(sol[1]), "free_prop_example.mp4");
nothing #hide
````

![](free_prop_example.mp4)

## Gross-Pitaevskii Equation

We will now add a nonlinear term to the equation, which will make it a proper Gross-Pitaevskii equation: i ∂u(r, t)/∂t = -∇²u / 2 + g|u|²u
where g is a coupling constant that defines the strength of the nonlinearity.

To add this nonlinear term, we need to define a new function that takes the wavefunction and returns the nonlinearity term
(only the g|u|² part. Check [`GrossPitaevskiiProblem`](@ref) for more information.)

````@example quick_start
nonlinearity(u, param) = param.g * abs2(u[1]);
nothing #hide
````

Here, we have used the `param` argument to pass the coupling constant `g`.
We define this argument as a named tuple:

````@example quick_start
param = (; g=-6);
nothing #hide
````

The negative value of `g` indicates that we have an attractive nonlinearity, which will be clear in the visualization.
Now we can create a new problem instance with the nonlinearity:

````@example quick_start
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity, param);
nothing #hide
````

We used the same initial condition, lengths, and dispersion as before, but now we have added the nonlinearity and the parameters.

Just as before, we can solve the problem and visualize the solution:

````@example quick_start
tspan = (0, 0.4)
ts, sol = solve(prob, alg, tspan; nsaves, dt, show_progress=false);

save_animation(abs2.(sol[1]), "gross_pitaevskii.mp4");
nothing #hide
````

The attractive nonlinearity will cause the wavefunction to collapse into a localized structure, which can be seen in the animation.

![](gross_pitaevskii.mp4)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

