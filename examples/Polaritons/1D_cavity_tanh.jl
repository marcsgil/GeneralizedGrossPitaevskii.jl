using Revise, GeneralizedGrossPitaevskii, CairoMakie, LinearAlgebra, CUDA, Statistics, ProgressMeter, KernelAbstractions, FFTW
include("polariton_funcs.jl")

function dispersion(ks, param)
    -im * param.γ / 2 + param.ħ * sum(abs2, ks) / 2param.m - param.δ₀
end

function amplitude(t, Amax, t_cycle, t_freeze)
    _t = ifelse(t > t_freeze, t_freeze, t)
    val = Amax * _t * (t_cycle - _t) * 4 / t_cycle^2
    val < 0 ? zero(val) : val
end

function potential(rs, param)
    param.V_def * exp(-sum(abs2, rs) / param.w_def^2) +
    param.V_damp * damping_potential(rs, -param.L / 2, param.L / 2, param.w_damp)
end

function phase(x, width, k_up, k_down)
    k₋ = (k_down - k_up) / 2
    k₊ = (k_down + k_up) / 2
    k₋ * width * log(cosh(x / width)) + k₊ * x
end

function pump(x, param, t)
    a = amplitude(t, param.Amax, param.t_cycle, param.t_freeze)
    if x[1] ≤ -0.9 * param.waist
        a *= 4
    elseif x[1] ≥ 0
        a *= 1
    end
    ϕ = phase(x[1], param.width, param.k_up, param.k_down)
    a * (abs(x[1]) ≤ param.waist) * cis(ϕ)
end

noise_func(ψ, param) = √(param.γ / 2 / param.δL)

nonlinearity(ψ, param) = param.g * abs2(ψ)

# Space parameters
L = 800.0f0
lengths = (L,)
N = 1024
δL = L / N
rs = range(; start=-L / 2, step=L / N, length=N)

# Polariton parameters
ħ = 0.6582f0 #meV.ps
γ = 0.047f0 / ħ
m = 7 / 16f0
g = 0.0003f0 / ħ
δ₀ = 0.2 / ħ

# Potential parameters
V_damp = 100.0f0
w_damp = 5.0f0
V_def = -0.85f0 / ħ
w_def = 0.75f0

# Pump parameters
k_up = 0.1f0
k_down = 0.55f0

δ_up = δ₀ - ħ * k_up^2 / 2m
δ_down = δ₀ - ħ * k_down^2 / 2m

waist = 300f0
width = 9f0

# Bistability cycle parameters
Imax = 90.0f0
Amax = √Imax
t_cycle = 1000.0f0
t_freeze = 974.0f0

δt = 5f-2

# Full parameter tuple
param = (; δ₀, m, γ, ħ, L, g,
    Amax, t_cycle, t_freeze, δL, waist, width, k_up, k_down,
    V_damp, w_damp, V_def, w_def)

u0_empty = CUDA.zeros(ComplexF32, N)
prob_steady = GrossPitaevskiiProblem(u0_empty, lengths; dispersion, nonlinearity, pump, param)
tspan_steady = (0, 2000.0f0)
solver_steady = StrangSplittingC(512, δt)
ts_steady, sol_steady = solve(prob_steady, solver_steady, tspan_steady);

steady_state = sol_steady[:, end]
heatmap(rs, ts_steady, Array(abs2.(sol_steady)))
##
with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"gn")
    offset = 300
    J = N÷2-offset:N÷2+offset
    lines!(ax, rs[J], g * Array(abs2.(steady_state[J])), linewidth=4)
    fig
end
##
ks = GeneralizedGrossPitaevskii.reciprocal_grid(prob_steady)[1]

ϕ₊ = angle.(steady_state[2:end])
ϕ₋ = angle.(steady_state[1:end-1])
∇ϕ = mod2pi.(ϕ₊ - ϕ₋) / δL
v = ħ * ∇ϕ / m

ψ₀ = steady_state[2:end-1]
ψ₊ = steady_state[3:end]
ψ₋ = steady_state[1:end-2]
∇ψ = (ψ₊ + ψ₋ - 2ψ₀) / param.δL^2
δ_vec = δ₀ .+ ħ * real(∇ψ ./ ψ₀) / 2m

c = [speed_of_sound(abs2(ψ), δ, g, ħ, m) for (ψ, δ) ∈ zip(Array(steady_state[2:end-1]), Array(δ_vec))]

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x")
    offset = 300
    J = N÷2-offset:N÷2+offset
    lines!(ax, rs[J], c[J], linewidth=4, color=:blue, label=L"c")
    lines!(ax, rs[J], Array(v[J]), linewidth=4, color=:red, label=L"v")
    axislegend(; position=:rb)
    fig
end
##
ns_theo = LinRange(0, 800, 512)
Is_theo = eq_of_state.(ns_theo, δ_up, g, γ)

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=16)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(ax, Is_theo, ns_theo, color=:blue, linewidth=4, label="Theoretical")
    A_stop = amplitude(Inf, param.Amax, param.t_cycle, param.t_freeze)
    scatter!(ax, abs2(A_stop), abs2(Array(steady_state)[N÷2 - 50, end]), color=:black, markersize=16)
    fig
end
##
using ForwardDiff, Roots

function get_extrema(up_bracket, down_bracket, param_up, param_down)

    f(k) = ForwardDiff.derivative(k -> dispersion_relation(k, param_up...), k)
    k_min = find_zero(f, up_bracket, Bisection())

    g(k) = ForwardDiff.derivative(k -> dispersion_relation(k, param_down...), k)
    k_max = find_zero(g, down_bracket, Bisection())

    k_min, dispersion_relation(k_min, param_up...), k_max, dispersion_relation(k_max, param_down...)
end

up_bracket = -1, 1
down_bracket = -1.5, 1.5
ks_up = LinRange(up_bracket..., 512)
ks_down = LinRange(down_bracket..., 512)

n_up = abs2(Array(steady_state)[N÷2-50])

param_up = (k_up, g, n_up, δ_up, m)
ω₊_up = dispersion_relation.(ks_up, param_up..., true)

n_down = abs2(Array(steady_state)[N÷2+50])

param_down = (k_down, g, n_down, δ_down, m)
ω₊_down = dispersion_relation.(ks_down, param_down..., true)
ω₋_down = dispersion_relation.(ks_down, param_down..., false)

k_min, ω_min, k_max, ω_max = get_extrema(up_bracket, down_bracket, (param_up..., true), (param_down..., false))

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(800, 400))
    ax1 = Axis(fig[1, 1]; xlabel=L"k", ylabel=L"\omega")
    ax2 = Axis(fig[1, 2]; xlabel=L"k")

    ylims!(ax1, -0.2, 0.5)
    ylims!(ax2, -0.2, 0.5)

    hideydecorations!(ax2, grid=false)
    colgap!(fig.layout, 0)

    lines!(ax1, ks_up, ω₊_up, linewidth=4)
    lines!(ax2, ks_down, ω₊_down, linewidth=4)
    lines!(ax2, ks_down, ω₋_down, linewidth=4)

    for ax ∈ (ax1, ax2)
        hlines!(ax, ω_min, linewidth=4, linestyle=:dash, color=:green)
        hlines!(ax, ω_max, linewidth=4, linestyle=:dash, color=:green)
    end

    fig
end
##
ωs = LinRange(ω_min, ω_max, 512)

half_up_bracket = (up_bracket[1], k_min)
half_down_bracket = (down_bracket[1], k_max)

corr_up = [find_zero(k -> dispersion_relation(k, param_up..., true) - ω, half_up_bracket, Bisection()) for ω ∈ ωs]
corr_down = [find_zero(k -> dispersion_relation(k, param_down..., false) - ω, half_down_bracket, Bisection()) for ω ∈ ωs]

lines(corr_up, corr_down, linewidth=4)
##
function one_point_corr!(dest, sol)
    backend = get_backend(dest)

    @kernel function kernel!(dest, sol)
        j = @index(Global)
        x = zero(eltype(dest))
        for k ∈ axes(sol, 2)
            x += abs2(sol[j, k])
        end
        dest[j] += x
    end

    kernel!(backend, 64)(dest, sol, ndrange=length(dest))
    KernelAbstractions.synchronize(backend)
end

function two_point_corr!(dest, sol)
    backend = get_backend(dest)

    @kernel function kernel!(dest, sol)
        j, k = @index(Global, NTuple)
        x = zero(eltype(dest))
        for m ∈ axes(sol, 2)
            x += abs2(sol[j, m]) * abs2(sol[k, m])
        end
        dest[j, k] += x
    end

    kernel!(backend, 64)(dest, sol, ndrange=size(dest))
    KernelAbstractions.synchronize(backend)
end

function calculate_correlation(steady_state, lengths, batchsize, nbatches, tspan, δt; param, kwargs...)
    u0_steady = stack(steady_state for _ ∈ 1:batchsize)
    noise_prototype = similar(u0_steady)

    prob = GrossPitaevskiiProblem(u0_steady, lengths; noise_prototype, param, kwargs...)
    solver = StrangSplittingC(1, δt)

    one_point = similar(steady_state, real(eltype(steady_state)))
    two_point = similar(steady_state, real(eltype(steady_state)), size(steady_state, 1), size(steady_state, 1))

    fill!(one_point, 0)
    fill!(two_point, 0)

    for batch ∈ 1:nbatches
        @info "Batch $batch"
        ts, _sol = solve(prob, solver, tspan; save_start=false)
        sol = dropdims(_sol, dims=3)
        ft_sol = fftshift(fft(ifftshift(sol, 1), 1), 1)

        one_point_corr!(one_point, sol)
        two_point_corr!(two_point, sol)
    end

    one_point /= nbatches * batchsize
    two_point /= nbatches * batchsize

    δ = one(two_point)
    factor = 1 / 2param.δL
    n = one_point .- factor

    (two_point .- factor .* (1 .+ δ) .* (n .+ n' .+ factor)) ./ (n .* n')
end

tspan_noise = (0.0f0, 50.0f0) .+ tspan_steady[end]

G2 = calculate_correlation(steady_state, lengths, 10^5, 1, tspan_noise, δt;
    dispersion, nonlinearity, pump, param, noise_func)

J = N÷2-130:N÷2+130
ks = range(; start=-π / δL, step=2π / (N * δL), length=N)
##
with_theme(theme_latexfonts()) do
    fig = Figure(; size=(730, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"x", ylabel=L"x\prime")
    hm = heatmap!(ax, rs[J], rs[J], (Array(real(G2)[J, J]) .- 1) * 1e5, colorrange=(-5, 5), colormap=:inferno)
    #hm = heatmap!(ax, ks[J], ks[J], (Array(real(G2)[J, J]) .- 1) * 8e3, colorrange=(-5, 5), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(x, x\prime) -1 \ \ ( \times 10^{-5})")
    #lines!(ax, corr_up, corr_down, linewidth=4)
    #save("dev_env/g2m1.pdf", fig)
    fig
end
##