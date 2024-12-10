using GeneralizedGrossPitaevskii, CairoMakie, StructuredLight

L = 800.0f0
lengths = (L,)
tspan = (0, 5)
N = 512
rs = range(; start=-L / 2, length=N, step=L / N)
#u0 = ComplexF32[exp(-x^2 - y^2) for x in rs, y in rs]
u0 = zeros(ComplexF32, N,)

dispersion(ks, param) = sum(abs2, ks) / 2 - im
potential(x, param) = 1 * damping_potential(x, -10f0, 10f0, 0.01)
g = 1f0
pump(x, param, t) = first(x) ≤ 0

prob = GrossPitaevskiiProblem(u0, lengths, dispersion, potential, g, pump)
solver = StepSplitting(128, 2.0f-4)

ts, sol = solve(prob, solver, tspan)

heatmap(abs2.(sol))
##
fig = Figure()
ax = Axis(fig[1,1])
#ylims!(ax, (0, 1.2))
lines!(ax, abs2.(sol[:,end]))
fig