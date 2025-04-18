using GeneralizedGrossPitaevskii

N = 64
L = rand(4.0f0:1.0f0:10.0f0)
lengths = (L, L)
ΔL = L / N
dt = 0.3f0 * rand(Float32)
nsaves = rand(50:200)

rs = range(; start=-L / 2, length=N, step=ΔL)
u0 = (randn(ComplexF64, N, N),)

dispersion(ks, param) = sum(abs2, ks) / 2
prob = GrossPitaevskiiProblem(u0, lengths; dispersion)

tspan = (0, nsaves * dt)

alg = StrangSplitting()

ts, sol = solve(prob, alg, tspan; nsaves, dt, show_progress=false)
new_dispersion(args...) = SVector(dispersion(args...))

prob2 = GrossPitaevskiiProblem(u0, lengths; dispersion=new_dispersion)
ts, sol = solve(prob2, alg, tspan; nsaves, dt, show_progress=false)