using Revise, BenchmarkTools
using GeneralizedGrossPitaevskii
using CairoMakie, StructuredLight

const ħ = 0.6582f0 #meV.ps
const L = 800.0f0

function dispersion(ks, param)
    δ₀, m, γ = param
    -im * γ / 2 + ħ * sum(abs2, ks) / 2m - δ₀
end

function potential(rs, param)
    100 * damping_potential(rs, -L / 2, L / 2, 10) #- 0.85 / ħ * exp(-sum(abs2, rs) / 0.75^2)
end

function I(t, t_cycle, Imax)
    val = -Imax * t * (t - t_cycle) * 4 / t_cycle^2
    val < 0 ? zero(val) : val
end

function plato_pump(x, xmin, xmax, width)
    x̅ = (xmin + xmax) / 2
    if x ≤ x̅
        1 + tanh((x - xmin) / width)
    else
        1 - tanh((x - xmax) / width)
    end
end

function bistability_cycle(g, δ₀, m, γ, kₚ, L, N, Imax, t_cycle, t_stop, t_end, solver)
    lengths = (L,)

    u0 = zeros(ComplexF32, ntuple(n -> N, length(lengths)))

    function pump(x, param, t)
        δ₀, m, γ, kₚ, t_cycle, t_stop, Imax = param
        _t = t > t_stop ? t_stop : t
        (x[1] ≤ -7) * √I(_t, t_cycle, Imax) * cis(kₚ * x[1])
        #(abs(x[1]) ≤ 200) * √I(t, t_cycle, Imax) #* cis(kₚ * x[1])
        #exp(-sum(abs2,x) / 200^2) * √I(t, t_cycle, Imax) * cis(-kₚ * x[1])
    end

    param = (δ₀, m, γ, kₚ, t_cycle, t_stop, Imax)
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

N = 512

rs = range(; start=-L / 2, step=L / N, length=N)

Imax = 90.0f0
t_cycle = 300
tstop = 285
t_end = 1000
solver = StrangSplitting(1024, 2.0f-2)

ts, sol = bistability_cycle(g, δ₀, m, γ, kₚ, L, N, Imax, t_cycle, tstop, t_end, solver)
heatmap(rs, ts, (abs2.(sol)))
##

_ts = filter(t -> t ≤ tstop, ts)

Is = I.(_ts, t_cycle, Imax)
color = [n ≤ length(ts) / 2 ? :red : :black for n ∈ eachindex(ts)]

function bistability_curve(n, δ, g, γ)
    n * (γ^2 / 4 + (g * n - δ)^2)
end

ns_theo = LinRange(0, 1800, 512)
Is_theo = [bistability_curve(n, δ, g, γ) for n ∈ ns_theo]
ns = abs2.(sol[N÷4, :])
#ns = abs2.(sol[N÷2, :])

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=24)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(Is, ns[1:length(Is)]; label="Simulation", color=:red, linewidth=5)
    lines!(ax, Is_theo, ns_theo, color=:blue, linewidth=5, label="Theory", linestyle=:dash)
    axislegend(ax, position=:lt)
    #save("bistability.png", fig)
    fig
end
##
function speed_of_sound(g, n₀, δ, m)
    √(ħ * (2 * g * n₀ - δ) / m)
end

function dispersion_relation(k, kₚ, g, n₀, δ, m, branch::Bool)
    cₛ = speed_of_sound(g, n₀, δ, m)
    v = ħ * kₚ / m
    pm = branch ? 1 : -1
    gn₀ = g * n₀
    v * k + pm * √(ħ^2 * k^4 / 4m^2 + (cₛ * k)^2 + (gn₀ - δ) * (3gn₀ - δ))
end

ks = LinRange(-1, 1, 512)

ωs₊ = dispersion_relation.(ks, kₚ, g, ns[end], δ, m, true)
ωs₋ = dispersion_relation.(ks, kₚ, g, ns[end], δ, m, false)

fig = Figure()
ax = Axis(fig[1, 1]; xlabel=L"k", ylabel=L"\omega")
#ylims!(-1, 1)
lines!(ax, ks, ωs₊)
lines!(ax, ks, ωs₋)
fig