using Revise, BenchmarkTools
using GeneralizedGrossPitaevskii
using CairoMakie, StructuredLight

function bistability_curve(n, δ, g, γ)
    n * (γ^2 / 4 + (g * n - δ)^2)
end

ns_theo = LinRange(0, 41, 512)

ω₀ = 1483.0f0
g = 0.01f0
δ = 0.3f0
ωₚ = ω₀ + δ
kz = 27.0f0
γ = 0.1f0

Is_theo = bistability_curve.(ns_theo, δ, g, γ)

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=16)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(ax, Is_theo, ns_theo, color=:blue, linewidth=4, label="Theoretical")
    fig
end
##
function dispersion(ks, param)
    tmax, Imax, width, ωₚ, ω₀, kz, γ = param
    -im * γ / 2 + ω₀ * (1 + sum(abs2, ks) / 2kz^2) - ωₚ
end

potential = nothing

function I(t, tmax, Imax)
    val = -Imax * t * (t - tmax) * 4 / tmax^2
    val < 0 ? zero(val) : val
end

function pump(x, param, t)
    tmax, Imax, width = param
    exp(-sum(abs2, x) / width^2) * √I(t, tmax, Imax)
end

L = 256.0f0
lengths = (L,)
u0 = zeros(ComplexF32, ntuple(n -> 256, length(lengths)))

Imax = maximum(Is_theo)
width = 50.0f0
##
δt = 0.1f0
nsaves = 1024
tspan = (0, 4000)
tmax = tspan[end]

param = (tmax, Imax, width, ωₚ, ω₀, kz, γ)

prob = GrossPitaevskiiProblem(u0, lengths, dispersion, potential, g, pump, param)
solver = StrangSplitting(nsaves, δt)

ts, sol = solve(prob, solver, tspan)
heatmap(abs2.(sol))
##
Is = I.(ts, tmax, Imax)
color = [n ≤ length(ts) / 2 ? :red : :black for n ∈ eachindex(ts)]

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=24)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(Is, dropdims(maximum(abs2, sol, dims=1), dims=1); label="Simulation", color=:red, linewidth=5)
    lines!(ax, Is_theo, ns_theo, color=:blue, linewidth=5, label="Theory", linestyle=:dash)
    axislegend(ax, position=:rb)
    #save("bistability.png", fig)
    fig
end
