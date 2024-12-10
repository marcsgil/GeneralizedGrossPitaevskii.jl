using Revise, BenchmarkTools
using GeneralizedGrossPitaevskii
using CairoMakie, StructuredLight

const ħ = 0.6582f0 #meV.ps

function dispersion(ks, param)
    δ₀, m, γ = param
    -im * γ / 2 + ħ * sum(abs2, ks) / 2m - δ₀
end

function I(t, t_cycle, Imax)
    val = -Imax * t * (t - t_cycle) * 4 / t_cycle^2
    val < 0 ? zero(val) : val
end

function bistability_cycle(g, δ₀, m, γ, kₚ, L, N, Imax, t_cycle, t_end, solver)
    lengths = (L,)

    u0 = zeros(ComplexF32, ntuple(n -> N, length(lengths)))

    function potential(rs, param)
        100 * damping_potential(rs, -L / 2, L / 2, 1)
    end

    function pump(x, param, t)
        δ₀, m, γ, kₚ, t_cycle, Imax = param
        (x[1] ≤ 0) * √I(t, t_cycle, Imax) * cis(kₚ * x[1])
    end

    param = (δ₀, m, γ, kₚ, t_cycle, Imax)
    prob = GrossPitaevskiiProblem(u0, lengths, dispersion, potential, g, pump, param)

    tspan = (0, t_end)
    solve(prob, solver, tspan)
end

ω₀ = 1473.36f0 / ħ
ωₚ = 1473.85 / ħ
kₚ = 0.27f0

γ = 0.047f0 / ħ
g = 0.0003f0 / ħ
m = ħ^2 / (2 * 1.29f0)

δ₀ = ωₚ - ω₀
δ = δ₀ - ħ * kₚ^2 / 2m

L = 800.0f0
N = 256

rs = range(; start=-L / 2, step=L / N, length=N)

Imax = 90.0f0
t_cycle = 3400
t_end = 3400
solver = StrangSplitting(1024, 1f0/2)

ts, sol = bistability_cycle(g, δ₀, m, γ, kₚ, L, N, Imax, t_cycle, t_end, solver)
heatmap(rs, ts, abs2.(sol))

ts
##
Is = I.(ts, t_cycle, Imax)
color = [n ≤ length(ts) / 2 ? :red : :black for n ∈ eachindex(ts)]

function bistability_curve(n, δ, g, γ)
    n * (γ^2 / 4 + (g * n - δ)^2)
end

ns_theo = LinRange(0, 1800, 512)
Is_theo = [bistability_curve(n, δ, g, γ) for n ∈ ns_theo]
ns = abs2.(sol[N÷4, :])

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=24)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(Is, ns; label="Simulation", color=:red, linewidth=5)
    lines!(ax, Is_theo, ns_theo, color=:blue, linewidth=5, label="Theory", linestyle=:dash)
    axislegend(ax, position=:rb)
    #save("bistability.png", fig)
    fig
end
##
ts[end-2] - ts[end-3]