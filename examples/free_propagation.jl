using GeneralizedGrossPitaevskii, StructuredLight, CairoMakie

N = 256
L = 10.0f0
ΔL = L / N
δt = 0.01f0
lengths = (L, L)

rs = range(; start=-L / 2, length=N, step=ΔL)
u0 = lg(rs, rs, l=1) + lg(rs, rs, l=-2)

visualize(abs2.(u0))

dispersion(ks, param) = sum(abs2, ks) / 2

prob = GrossPitaevskiiProblem(u0, lengths; dispersion)
solver = StrangSplittingB(100, δt)
ts, sol = solve(prob, solver, (0, 1))

save_animation(abs2.(sol), "free_prop_example.mp4")