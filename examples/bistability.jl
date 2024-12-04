using Revise, BenchmarkTools
using GeneralizedGrossPitaevskii
using CairoMakie

function bistability_curve(n, δ, g, γ)
    n * (γ^2 / 4 + (g * n - δ)^2)
end

ns_theo = LinRange(0, 43, 512)

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
function dispersion!(dest, ks...; param)
    #tmax, Imax, width, ωₚ, ω₀, kz, γ = param
    #dest[1] = -γ / 2 + im * (ωₚ - ω₀ * (1 + sum(abs2, ks) / 2kz^2))
    dest[1] = sum(abs2, ks) / 2
end

potential! = nothing

nonlinearity = reshape([g], 1, 1)

function I(t, tmax, Imax)
    -Imax * t * (t - tmax) * 4 / tmax^2
end

function pump!(dest, x; param)
    t, tmax, Imax, width = param
    dest[1] = (abs(x) ≤ width / 2) * I(t, tmax, Imax)
end

L = 256.0f0
lengths = (L,)
u₀ = zeros(ComplexF32, 1, 256)

tmax = 2000
Imax = maximum(Is_theo)
width = 80.0f0

param = (tmax, Imax, width, ωₚ, ω₀, kz, γ)

prob = GrossPitaevskiiProblem(dispersion!, potential!, nonlinearity, nothing, u₀, lengths)
##
δt = 0.01f0
nsteps = round(Int, tmax / δt)
nsaves = 512

prob.ks

prob.spatial_dims

sol = solve(prob, StrangSplitting(), nsteps, nsaves, δt)
##
sol = dropdims(sol, dims=1)

heatmap(abs2.(sol))
##

Is = dropdims(maximum(abs2, sol, dims=1), dims=1)

lines(Is)