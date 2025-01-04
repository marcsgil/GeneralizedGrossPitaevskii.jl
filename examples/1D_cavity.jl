using GeneralizedGrossPitaevskii, CairoMakie, FFTW

function dispersion(ks, param)
    -im * param.γ / 2 + param.ħ * sum(abs2, ks) / 2 / param.m - param.δ₀
end

function potential(rs, param)
    param.V_damp * damping_potential(rs, -param.L / 2, param.L / 2, param.w_damp) + param.V_def * exp(-sum(abs2, rs) / param.w_def^2)
end

function A(t, Amax, t_cycle, t_freeze)
    _t = ifelse(t > t_freeze, t_freeze, t)
    val = Amax * _t * (t_cycle - _t) * 4 / t_cycle^2
    val < 0 ? zero(val) : val
end

function pump(x, param, t)
    (x[1] ≤ -7) * A(t, param.Amax, param.t_cycle, param.t_freeze) * (1 + 3 * (x[1] ≤ (-param.L / 2 + 10))) * cis(x[1] * param.k_pump)
end

# Space parameters
L = 1800.0f0
lengths = (L,)
N = 1024
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
param = (; δ₀, m, γ, ħ, L, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, t_freeze, k_pump)

u0 = zeros(ComplexF32, ntuple(n -> N, length(lengths)))
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, potential, nonlinearity=g, pump, param)
tspan = (0, 1000.0f0)
solver = StrangSplittingB(4096, 4.0f-1)
ts, sol = solve(prob, solver, tspan)

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="t")
    heatmap!(ax, rs, ts, Array(abs2.(sol)))
    fig
end
##
Is = @. A(ts, Amax, t_cycle, t_freeze)^2

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
    axislegend(ax, position=:lt)
    fig
end
##
function dispersion_relation(k, kₚ, g, n₀, δ, m, branch::Bool)
    v = ħ * kₚ / m
    pm = branch ? 1 : -1
    gn₀ = g * n₀
    v * k +pm * √(ħ^2 * k^4 / 4m^2 + (ħ * (2 * g * n₀ - δ) / m) * k^2 + (gn₀ - δ) * (3gn₀ - δ))
end


u0_steady = sol[:, end]

J = 100:220

δψ = (sol[J, 1800:end] .- u0_steady[J]) .* cis.(-k_pump .* rs[J])
heatmap(abs2.(δψ))
##
Δt = ts[2] - ts[1]
Δx = rs[2] - rs[1]

Nx = size(δψ, 1)
Nt = size(δψ, 2)

ks = range(; start=-π / Δx, step=2π / (Nx * Δx), length=Nx)
ωs = range(; start=-π / Δt, step=2π / (Nt * Δt), length=Nt)


log_δψ̃ = δψ |> fftshift |> fft |> ifftshift .|> abs .|> log
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
J = 700:820
δψ = (sol[J, 1800:end] .- u0_steady[J]) .* cis.(-k_pump .* rs[J])
heatmap(abs2.(δψ))


Δt = ts[2] - ts[1]
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
    ω₊ = dispersion_relation.(ks, k_pump, g, 0, δ, m, true)
    ω₋ = dispersion_relation.(ks, k_pump, g, 0, δ, m, false)
    heatmap!(ax, ks, ωs, log_δψ̃, colormap=:magma)
    lines!(ax, ks, ω₊, color=:red, linestyle=:dot, linewidth=4)
    lines!(ax, ks, ω₋, color=:blue, linestyle=:dot, linewidth=4)
    fig
end