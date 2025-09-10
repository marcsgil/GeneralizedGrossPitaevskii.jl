```@meta
EditURL = "../../examples/truncated_wigner.jl"
```

````@example truncated_wigner
using GeneralizedGrossPitaevskii, FFTW, CairoMakie

function dispersion(ks, param)
    param.ħ * sum(abs2, ks) / 2param.m - param.δ - im * param.γ / 2
end

function pump(x, param, t)
    param.A
end

nonlinearity(ψ, param) = param.g * (abs2(first(ψ)) - 1 / param.dx)

position_noise_func(ψ, xs, param) = √(param.γ / 2param.dx)

ħ = 0.6582 #meV.ps
γ = 0.047 / ħ
m = 1 / 6 # meV.ps^2/μm^2; This is 3×10^-5 the electron mass
g = 3e-4 / ħ
δ = 0.49 / ħ
A = 10

L = 512
N = 256
dx = L / N

lengths = (L,)

param = (; ħ, m, δ, γ, g, A, L, dx)
u0 = (zeros(ComplexF64, N),);
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity, pump, param)
nsaves = 512
dt = 0.05
tspan = (0, 200)
alg = StrangSplitting()
ts, sol = solve(prob, alg, tspan; nsaves, dt);

ns = abs2.(sol[1][1, :])

with_theme(theme_latexfonts()) do
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="t (ps)", ylabel="n (μm⁻¹)")
    lines!(ax, ts, ns; colormap=:viridis, colorrange=(0, 4))
    fig
end
#
ns_theo = LinRange(0, 2200, 512)
Is = @. ((g * ns_theo - δ)^2 + (γ / 2)^2) * ns_theo
n = ns[end]
with_theme(theme_latexfonts()) do
    fig = Figure()
    ax = Axis(fig[1, 1]; ylabel="n (μm⁻¹)", xlabel="I (meV.μm⁻¹.ps⁻¹)")
    lines!(ax, Is, ns_theo; colormap=:viridis, colorrange=(0, 4))
    scatter!(ax, [A^2], [n]; color=:red, markersize=10, label="Truncated Wigner")
    fig
end
#
u0 = (randn(ComplexF64, N, 512) / 2dx,);
noise_prototype = similar.(u0);
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity, pump, param, noise_prototype, position_noise_func)
ts, sol = solve(prob, alg, tspan; nsaves=1, dt, save_start=false)

ft_sol = fftshift(fft(sol[1], 1), 1) / N
ks = fftshift(fftfreq(N, 2π / dx))

nks = dropdims(mean(abs2, ft_sol; dims=2), dims=(2, 3)) .- 1 / 2L
nks[N÷2+1] = NaN

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=16, size=(800, 400))
    ax1 = Axis(fig[1, 1]; ylabel="n", xlabel="I")
    lines!(ax1, Is, ns_theo; colormap=:viridis, colorrange=(0, 4))
    scatter!(ax1, [A^2], [n]; color=:red, markersize=10, label="Truncated Wigner")
    ax2 = Axis(fig[1, 2]; xlabel="k (μm⁻¹)", ylabel="n(k)")
    lines!(ax2, ks, nks, linewidth=2)
    fig
end
#
function f(x, y, δ, commutator)
    abs2(x) * abs2(y) - (1 + δ) * commutator / 2 * (abs2(x) + abs2(y) - commutator / 2)
end

function G2(sol, commutator)
    result = similar(sol, real(eltype(sol)), (size(sol, 1), size(sol, 1)))

    for n ∈ axes(result, 2), m in axes(result, 1)
        result[m, n] = mapreduce((x, y) -> f(x, y, m == n, commutator), +,
            view(sol, m, :), view(sol, n, :)) / size(sol, 2)
    end

    result
end

ft_sol


G2k = G2(ft_sol, 1 / L)
#G2k[N÷2+1, N÷2+1] = NaN
nks * nks'
g2k = G2k ./ (nks * nks')

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=16, size=(500, 400))
    ax = Axis(fig[1, 1])
    xlims!(ax, -0.8, 0.8)
    ylims!(ax, -0.8, 0.8)
    hm = heatmap!(ax, ks, ks, g2k, colorrange=(0, 3))
    Colorbar(fig[1, 2], hm)
    fig
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

