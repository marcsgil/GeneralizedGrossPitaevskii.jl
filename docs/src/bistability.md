```@meta
EditURL = "../../examples/bistability.jl"
```

````@example bistability
using GeneralizedGrossPitaevskii, CairoMakie

function bistability_curve(n, δ, g, γ)
    n * (γ^2 / 4 + (g * n - δ)^2)
end

ns_theo = LinRange(0, 41, 512)

ω₀ = 1483.0
g = 0.01
δ = 0.3
kz = 27.0
γ = 0.1

Is_theo = bistability_curve.(ns_theo, δ, g, γ)

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=16)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(ax, Is_theo, ns_theo, color=:blue, linewidth=4, label="Theoretical")
    fig
end
#
function dispersion(ks, param)
    -im * param.γ / 2 + param.ω₀ * sum(abs2, ks) / 2param.kz^2 - param.δ
end

function I(t, tmax, Imax)
    val = -Imax * t * (t - tmax) * 4 / tmax^2
    val < 0 ? zero(val) : val
end

function pump(x, param, t)
    exp(-sum(abs2, x .- param.L / 2) / param.width^2) * √I(t, param.tmax, param.Imax)
end

nonlinearity(ψ, param) = param.g * abs2(ψ[1])

L = 256
lengths = (L,)
N = 128
u0 = (zeros(ComplexF32, ntuple(n -> N, length(lengths))), )
xs = range(; start=-L / 2, step=L / N, length=N)

Imax = maximum(Is_theo)
width = 50.0
#
dt = 0.1
nsaves = 512
alg = StrangSplitting()

tspan = (0, 4000.0)
tmax = tspan[end]
param = (; tmax, Imax, width, δ, ω₀, kz, γ, g, L)

prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity, pump, param)

ts, sol = solve(prob, alg, tspan; dt, nsaves)
heatmap(abs2.(sol[1]))
#
Is = I.(ts, tmax, Imax)
color = [n ≤ length(ts) / 2 ? :red : :black for n ∈ eachindex(ts)]

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=24)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(Is, dropdims(maximum(abs2, sol[1], dims=1), dims=1); label="Simulation", color=:red, linewidth=5)
    lines!(ax, Is_theo, ns_theo, color=:blue, linewidth=5, label="Theory", linestyle=:dash)
    axislegend(ax, position=:rb)
    fig
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

