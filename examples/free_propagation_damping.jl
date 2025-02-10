using GeneralizedGrossPitaevskii, CairoMakie, StructuredLight

L = 5.0f0
lengths = (L, L)
tspan = (0, 1)
N = 512
rs = range(; start=-L / 2, length=N, step=L / N)
u0 = ComplexF32[exp(-x^2 - y^2) for x in rs, y in rs]

dispersion(ks, param) = sum(abs2, ks) / 2 - im

prob = GrossPitaevskiiProblem(u0, lengths; dispersion)
solver = StrangSplittingB(128, 2.0f-2)

ts, sol = solve(prob, solver, tspan)

save_animation(abs2.(sol), "free_prop_example_damp.mp4", share_colorrange=true)