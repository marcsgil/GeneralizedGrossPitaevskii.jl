using GeneralizedGrossPitaevskii, CairoMakie, StructuredLight

L = 5.0f0
lengths = (L, L)
tspan = (0, 1)
N = 512
rs = StepRangeLen(0f0, L / N, N)
u0 = (ComplexF32[exp(-(x - L / 2)^2 - (y - L / 2)^2) for x in rs, y in rs],)

dispersion(ks, param) = sum(abs2, ks) / 2 - im

prob = GrossPitaevskiiProblem(u0, lengths; dispersion)
dt = 2f-2
nsaves = 128
alg = StrangSplitting()

ts, sol = solve(prob, solver, tspan; dt, nsaves)

save_animation(abs2.(sol[1]), "free_prop_example_damp.mp4", share_colorrange=true)