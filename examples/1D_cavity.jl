using GeneralizedGrossPitaevskii, CairoMakie

kz = 27.0f0
γ = 1.0f-1
g = 1.0f-2

function dispersion(q, param)
    kz, γ = param
    sum(abs2, q) / 2kz - im * γ / 2
end

pump(x, param, t) = exp(-sum(abs2, x) / 50^2)

L = 256.0f0
lengths = (L,)
N = 512
xs = range(; start=-L / 2, step=L / N, length=N)
u0 = ComplexF32[exp(-x^2 / 50^2) for x in xs]
#u0 = zeros(ComplexF32, ntuple(n->N, length(lengths)))
param = (kz, 0)


prob = GrossPitaevskiiProblem(u0, lengths, dispersion, nothing, 1.0f-2, pump, param)

nsaves = 128
δt = 1.0f-1
tspan = (0, 2000)

solver = StrangSplitting(nsaves, δt)
t, sol = solve(prob, solver, tspan)

heatmap(abs2.(sol))