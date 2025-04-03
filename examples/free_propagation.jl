using GeneralizedGrossPitaevskii, StructuredLight, CairoMakie

N = 256
L = 10.0f0
ΔL = L / N
dt = 0.01f0
lengths = (L, L)

rs = StepRangeLen(0f0, L / N, N)
u0 = (lg(rs .- L / 2, rs .- L / 2, l=1) + lg(rs .- L / 2, rs .- L / 2, l=-2), )

visualize(abs2.(u0[1]))

dispersion(ks, param) = sum(abs2, ks) / 2

prob = GrossPitaevskiiProblem(u0, lengths; dispersion)
alg = StrangSplitting()
nsaves = 128
ts, sol = solve(prob, alg, (0, 1); nsaves, dt)

save_animation(abs2.(sol[1]), "free_prop_example.mp4")