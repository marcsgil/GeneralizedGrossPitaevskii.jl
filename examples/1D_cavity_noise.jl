using GeneralizedGrossPitaevskii, FFTW, CairoMakie

function dispersion(ks, param)
    δ, m, γ, ħ = param
    -im * γ / 2 + ħ * sum(abs2, ks) / 2m - δ
end

function potential(rs, param)
    δ, m, γ, ħ, L, V_damp, w_damp, V_def, w_def = param
    V_damp * damping_potential(rs, -L / 2, L / 2, w_damp) + V_def * exp(-sum(abs2, rs) / w_def^2)
end

function A(t, Amax, t_cycle, t_freeze)
    _t = ifelse(t > t_freeze, t_freeze, t)
    val = Amax * _t * (t_cycle - _t) * 4 / t_cycle^2
    val < 0 ? zero(val) : val
end

function pump_bistab(x, param, t)
    args..., L, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, t_freeze = param
    (x[1] ≤ -7) * A(t, Amax, t_cycle, t_freeze) * (1 + 3 * (x[1] ≤ (-L / 2 + 10)))
end

# Space parameters
L = 1800.0f0
lengths = (L,)
N = 512
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
k_pump = 0.27f0
δ₀ = ωₚ - ω₀
δ = δ₀ - ħ * k_pump^2 / 2m

# Bistability cycle parameters
Imax = 90.0f0
Amax = √Imax
t_cycle = 100.0f0
t_freeze = 95.0f0

# Full parameter tuple
param = (δ, m, γ, ħ, L, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, t_freeze)

u0_bistab = zeros(ComplexF32, ntuple(n -> N, length(lengths)))
prob_bistab = GrossPitaevskiiProblem(u0_bistab, lengths, dispersion, potential, g, pump_bistab, param)
tspan_bistab = (0, 500.0f0)
solver_bistab = StrangSplittingB(1024, 4.0f-1)
ts_bistab, sol_bistab = solve(prob_bistab, solver_bistab, tspan_bistab)

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="t")
    heatmap!(ax, rs, ts_bistab, (abs2.(sol_bistab)))
    fig
end
##
lines(abs2.(sol_bistab[:, end]))
##
Is = @. A(ts_bistab, Amax, t_cycle, t_freeze)^2
color = [n ≤ length(ts_bistab) / 2 ? :red : :black for n ∈ eachindex(ts_bistab)]

function bistability_curve(n, δ, g, γ)
    n * (γ^2 / 4 + (g * n - δ)^2)
end

ns_theo = LinRange(0, 1800, 512)
Is_theo = [bistability_curve(n, δ, g, γ) for n ∈ ns_theo]
ns = abs2.(sol_bistab[N÷4, :])

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=24)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(Is, ns; label="Simulation", color=:red, linewidth=5)
    lines!(ax, Is_theo, ns_theo, color=:blue, linewidth=5, label="Theory", linestyle=:dash)
    axislegend(ax, position=:lt)
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
    #v * k + pm * √(ħ^2 * k^4 / 4m^2 + (cₛ * k)^2 + (gn₀ - δ) * (3gn₀ - δ))
    +pm * √(ħ^2 * k^4 / 4m^2 + (cₛ * k)^2 + (gn₀ - δ) * (3gn₀ - δ))
end

function pump_steady(x, param, t)
    args..., L, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, t_freeze, η = param
    #F_pump = (x[1] ≤ -7) * A_steady * (1 + 3 * (x[1] ≤ (-L / 2 + 10)))
    #F_pump + randn(ComplexF32) * η * A_steady
    pump_bistab(x, param[begin:end-1], t) #+ randn(ComplexF32) * η * A_steady
end

u0_steady = sol_bistab[:, end]
A_steady = A(t_freeze, Amax, t_cycle, t_freeze)

η = 1f0

param_probe = (δ, m, γ, ħ, L, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, t_freeze, η)

prob = GrossPitaevskiiProblem(u0_steady, lengths, dispersion, potential, g, pump_steady, param_probe)
tspan = tspan_bistab[2] .+ (0, 340.0f0)
solver = StrangSplittingB(256, 4.0f-2)
ts_probe, sol_probe = solve(prob, solver, tspan)

heatmap(abs2.(sol_probe))
##
J = 100:220
δψ = sol_bistab[J, 900:end] .- u0_steady[J]
heatmap(abs2.(δψ))
##

Δt = ts_bistab[2] - ts_bistab[1]
Δx = rs[2] - rs[1]

Nx = size(δψ, 1)
Nt = size(δψ, 2)

ks = range(; start=-π / Δx, step=2π / (Nx * Δx), length=Nx)
ωs = range(; start=-π / Δt, step=2π / (Nt * Δt), length=Nt)


log_δψ̃ = δψ |> ifftshift |> fft |> fftshift .|> abs .|> log
reverse!(log_δψ̃, dims=2)
J = argmax(log_δψ̃)
log_δψ̃[J[1], :] .= min(log_δψ̃...)
log_δψ̃[:, J[2]] .= min(log_δψ̃...)

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1]; xlabel=L"k", ylabel=L"\omega")
    ω₊ = dispersion_relation.(ks, k_pump, g, ns[end], δ, m, true)
    ω₋ = dispersion_relation.(ks, k_pump, g, ns[end], δ, m, false)
    heatmap!(ax, ks, ωs, log_δψ̃, colormap=:magma)
    lines!(ax, ks, ω₊, color=:grey, linestyle=:dot, linewidth=4)
    lines!(ax, ks, ω₋, color=:grey, linestyle=:dot, linewidth=4)
    fig
end
##
J = 340:400
δψ = sol_probe[J, 900:1000] .- u0_steady[J]
heatmap(abs2.(δψ))
##

Δt = ts_probe[2] - ts_probe[1]
Δx = rs[2] - rs[1]

Nx = size(δψ, 1)
Nt = size(δψ, 2)

ks = range(; start=-π / Δx, step=2π / (Nx * Δx), length=Nx)
ωs = range(; start=-π / Δt, step=2π / (Nt * Δt), length=Nt)


log_δψ̃ = δψ |> ifftshift |> fft |> fftshift .|> abs .|> log
reverse!(log_δψ̃, dims=2)
J = argmax(log_δψ̃)
#log_δψ̃[J[1], :] .= min(log_δψ̃...)
#log_δψ̃[:, J[2]] .= min(log_δψ̃...)

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1]; xlabel=L"k", ylabel=L"\omega")
    ω₊ = dispersion_relation.(ks, 0, g, 0, δ₀, m, true)
    ω₋ = dispersion_relation.(ks, 0, g, 0, δ₀, m, false)
    heatmap!(ax, ks, ωs, log_δψ̃, colormap=:magma)
    lines!(ax, ks, ω₊, color=:grey, linestyle=:dot, linewidth=4)
    lines!(ax, ks, ω₋, color=:grey, linestyle=:dot, linewidth=4)
    fig
end