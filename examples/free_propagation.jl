using GeneralizedGrossPitaevskii, StructuredLight, CairoMakie

N = 256
L = 10.0f0
ΔL = L / N
δt = 0.01f0
nsteps = 100
lengths = (L, L)

rs = range(; start=-L / 2, length=N, step=ΔL)
ψ₀ = lg(rs, rs, l=1) + lg(rs, rs, l=-2)

visualize(abs2.(ψ₀))

D(ks, param) = sum(abs2, ks) / 2

prob = GrossPitaevskiiProblem(ψ₀, lengths, D, nothing, nothing, nothing, δt)
solver = StrangSplitting(100, δt)
solve(prob, solver, (0, 1))

typeof(prob)

typeof(prob) <: AbstractGrossPitaevskiiProblem{M,N,T,T1,T2,T3,T4,T5,T6,Nothing} where {M,N,T,T1,T2,T3,T4,T5,T6}

save_animation(abs2.(ψs), "examples/test.mp4")