using GeneralizedGrossPitaevskii, StructuredLight, CairoMakie

N = 256
L = 10.0f0
ΔL = L / N
δt = 0.01f0
nsteps = 100
lengths = (L, L)

rs = range(; start=-L / 2, length=N, step=ΔL)
ψ₀ = lg(rs, rs, l=1) + lg(rs,rs,l=-2)

visualize(abs2.(ψ₀))

A(kx, ky) = -im * (kx^2 + ky^2) / 2

prob = GrossPitaevskiiProblem(ψ₀, A, nothing, nothing, nothing, δt, lengths)

ψs = dropdims(solve(prob, nsteps, 1); dims=1)

save_animation(abs2.(ψs), "examples/test.mp4")