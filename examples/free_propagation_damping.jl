# # Damped Free Propagation

# In this example, we will simulate a damped free propagation.
# We will define a simple 2D system with a Gaussian initial condition and a damping term
# in the dispersion relation to simulate the damping effect.

# We first load the necessary packages.
using GeneralizedGrossPitaevskii, CairoMakie, StructuredLight

# Then, we define the parameters of the grid for our simulation and the initial condition.
L = 5
lengths = (L, L)
tspan = (0, 1)
N = 128
rs = StepRangeLen(0, L / N, N)
u0 = (ComplexF64[exp(-(x - L / 2)^2 - (y - L / 2)^2) for x in rs, y in rs],);

# Then, we define the dispersion relation with a damping term.
dispersion(ks, param) = sum(abs2, ks) / 2 - im;
# This corresponds to an equation of the form i ∂u(r, t)/∂t = -∇²u / 2 - i u, which includes a damping term due to the imaginary part.

# Now we can create the problem instance.
prob = GrossPitaevskiiProblem(u0, lengths; dispersion);

# Now, we define the solver parameters, get the solution and visualize the results.
dt = 0.02
nsaves = 64
alg = StrangSplitting()

ts, sol = solve(prob, alg, tspan; dt, nsaves, show_progress=false);

save_animation(abs2.(sol[1]), "free_prop_example_damp.mp4", share_colorrange=true);

# We can see that the wavefunction is damped over time, which is expected due to the damping term in the dispersion relation.

#md # ![](free_prop_example_damp.mp4)