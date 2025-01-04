using GeneralizedGrossPitaevskii, CairoMakie, FFTW, LinearAlgebra

function dispersion(ks, param)
    -im * param.γ / 2 + param.ħ * sum(abs2, ks) / 2 / param.m - param.δ₀
end

function potential(rs, param)
    (param.V_damp * damping_potential(rs, -param.L / 2, param.L / 2, param.w_damp)
     +
     param.V_def * exp(-sum(abs2, rs) / param.w_def^2))
end

function A(t, Amax, t_cycle, t_freeze)
    _t = ifelse(t > t_freeze, t_freeze, t)
    val = Amax * _t * (t_cycle - _t) * 4 / t_cycle^2
    val < 0 ? zero(val) : val
end

function pump(x, param, t)
    (x[1] ≤ -7) * A(t, param.Amax, param.t_cycle, param.t_freeze) * (1 + 3 * (x[1] ≤ (-param.L / 2 + 10))) * cis(x[1] * param.k_pump)
end

noise_func(ψ, param) = √(param.γ / 2 / param.δL)

# Space parameters
L = 800.0f0
lengths = (L,)
N = 512
δL = L / N
rs = range(; start=-L / 2, step=L / N, length=N)

# Polariton parameters
ħ = 0.6582f0 #meV.ps
ω₀ = 1473.36f0 / ħ
ωₚ = 1473.85f0 / ħ
γ = 0.047f0 / ħ
m = ħ^2 / (2 * 1.29f0)
g = 0.0003f0 / ħ

# Potential parameters
V_damp = 10.0f0
w_damp = 0.1f0
V_def = -0.85f0 / ħ
w_def = 0.75f0

# Pump parameters
k_pump = 0.25f0
δ₀ = ωₚ - ω₀
δ = δ₀ - ħ * k_pump^2 / 2m

# Bistability cycle parameters
Imax = 90.0f0
Amax = √Imax
t_cycle = 250.0f0
t_freeze = 240.0f0

# Full parameter tuple
param = (; δ₀, m, γ, ħ, L, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, t_freeze, δL, k_pump)

u0 = randn(ComplexF32, N, 10^4) / √(2δL)
#u0 = zeros(ComplexF32, N, 1)
noise_prototype = similar(u0)
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, potential, nonlinearity=g, pump, param, noise_func, noise_prototype)
tspan = (0, 300.0f0)
solver = StrangSplittingB(256, 3.0f-1)
ts, sol = solve(prob, solver, tspan)

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="t")
    heatmap!(ax, rs, ts, Array(abs2.(sol[:, 1, :])))
    fig
end
##
Is = @. A(ts, Amax, t_cycle, t_freeze)^2

function bistability_curve(n, δ, g, γ)
    n * (γ^2 / 4 + (g * n - δ)^2)
end

ns_theo = LinRange(0, 1800, 512)
Is_theo = [bistability_curve(n, δ, g, γ) for n ∈ ns_theo]
ns = abs2.(sol[N÷4, 1, :])

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=24)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(Is, ns; label="Simulation", color=:red, linewidth=5)
    lines!(ax, Is_theo, ns_theo, color=:blue, linewidth=5, label="Theory", linestyle=:dash)
    axislegend(ax, position=:lt)
    fig
end
##
steady = dropdims(mean(sol[:, :, end], dims=2), dims=2)

ψ = sol[:, :, end] .- steady

expval_n = dropdims(mean(x -> abs2(x) - 1 / 2δL, ψ, dims=2), dims=2)

J = 210:300

lines(rs[J], expval_n[J])
##
function G2(ψ, δL)
    result = similar(ψ, real(eltype(ψ)), (length(ψ), length(ψ)))
    for (n, Ψp) ∈ enumerate(ψ), (m, Ψ) ∈ enumerate(ψ)
        a2Ψ = abs2(Ψ)
        a2Ψp = abs2(Ψp)
        result[m, n] = a2Ψ * a2Ψp - (1 + (m == n)) / 2δL * (a2Ψ + a2Ψp - 1 / 2δL)
    end
    result
end

G2_val = mean(ψ -> G2(ψ, δL), eachslice(ψ, dims=2))
##
g2 = G2_val ./ (expval_n * expval_n')

g2[J, J]

extrema(g2)
heatmap(rs[J], rs[J], g2[J, J], colormap=:hot, colorrange=(-1,10))