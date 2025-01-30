using GeneralizedGrossPitaevskii, CairoMakie, LinearAlgebra, CUDA, Statistics, ProgressMeter, KernelAbstractions, FFTW, Revise, HDF5
include("polariton_funcs.jl")

function dispersion(ks, param)
    val = -im * param.γ / 2 + param.ħ * sum(abs2, ks) / 2param.m - param.δ₀
    SVector(val, -conj(val))
end

function potential(rs, param)
    val = param.V_def * exp(-sum(abs2, rs) / param.w_def^2) +
          param.V_damp * damping_potential(rs, -param.L / 2, param.L / 2, param.w_damp)
    SVector(val, -conj(val))
end

function A(t, Amax, t_cycle, t_freeze)
    _t = ifelse(t > t_freeze, t_freeze, t)
    val = Amax * _t * (t_cycle - _t) * 4 / t_cycle^2
    val < 0 ? zero(val) : val
end

function pump(x, param, t)
    a = A(t, param.Amax, param.t_cycle, param.t_freeze)
    if x[1] ≤ -param.L * 0.9 / 2 || x[1] ≥ -10
        a *= 0
    elseif -param.L * 0.9 / 2 < x[1] ≤ -param.L * 0.85 / 2
        a *= 6
    end
    val = a * cis(mapreduce(*, +, param.k_pump, x))
    SVector(val, conj(val))
end

function noise_func(ψ, param)
    val = √(im * param.g / param.δL)
    SVector(val * ψ[1], conj(val) * ψ[2])
end

function nonlinearity(ψ, param)
    val = param.g * prod(ψ)
    SVector(val, -val)
end

# Space parameters
L = 300.0f0
lengths = (L,)
N = 512
δL = L / N
rs = range(; start=-L / 2, step=L / N, length=N)

# Polariton parameters
ħ = 0.6582f0 #meV.ps
γ = 0.047f0 / ħ
m = ħ^2 / 2.5f0
g = 0.0003f0 / ħ
δ₀ = 0.49 / ħ

# Potential parameters
V_damp = 10.0f0
w_damp = 5.0f0
V_def = -0.85f0 / ħ
w_def = 0.75f0

# Pump parameters
k_pump = 0.25f0
δ = δ₀ - ħ * k_pump^2 / 2m

# Bistability cycle parameters
Imax = 90.0f0
Amax = √Imax
t_cycle = 300.0f0
t_freeze = 288.0f0

δt = 6.0f-2

# Full parameter tuple
param = (; δ₀, m, γ, ħ, L, g, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, t_freeze, δL, k_pump)

u0_empty = CUDA.fill(SVector{2,ComplexF32}(0, 0), N)
prob_steady = GrossPitaevskiiProblem(u0_empty, lengths; dispersion, potential, nonlinearity, pump, param)
tspan_steady = (0, 2000.0f0)
solver_steady = StrangSplittingC(512, δt)
ts_steady, sol_steady = solve(prob_steady, solver_steady, tspan_steady);

steady_state = sol_steady[:, end]
heatmap(rs, ts_steady, Array(abs2.(first.(sol_steady))))
##
with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"n")
    offset = 150
    J = N÷2-offset:N÷2+offset
    lines!(ax, rs[J], g * Array(abs2.(first.(steady_state[J]))), linewidth=4)
    fig
end
##
ks = GeneralizedGrossPitaevskii.reciprocal_grid(prob_steady)[1]

ϕ₊ = angle.(first.(steady_state[2:end]))
ϕ₋ = angle.(first.(steady_state[1:end-1]))
∇ϕ = mod2pi.(ϕ₊ - ϕ₋) / δL
v = ħ * ∇ϕ / m

ψ₀ = first.(steady_state[2:end-1])
ψ₊ = first.(steady_state[3:end])
ψ₋ = first.(steady_state[1:end-2])
∇ψ = (ψ₊ + ψ₋ - 2ψ₀) / param.δL^2
δ_vec = δ₀ .+ ħ * real(∇ψ ./ ψ₀) / 2m

c = [speed_of_sound(abs2(ψ), δ, g, ħ, m) for (ψ, δ) ∈ zip(Array(first.(steady_state[2:end-1])), Array(δ_vec))]

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x")
    offset = 100
    J = N÷2-offset:N÷2+offset
    lines!(ax, rs[J], c[J], linewidth=4, color=:blue, label=L"c")
    lines!(ax, rs[J], Array(v[J]), linewidth=4, color=:red, label=L"v")
    axislegend()
    fig
end
##
ns_theo = LinRange(0, 1500, 512)
Is_theo = eq_of_state.(ns_theo, δ, g, γ)

n_up = abs2(Array(first.(sol_steady))[N÷4, end])
n_down = abs2(Array(first.(sol_steady))[3N÷4, end])

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=16)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(ax, Is_theo, ns_theo, color=:blue, linewidth=4, label="Theoretical")
    A_stop = A(Inf, param.Amax, param.t_cycle, param.t_freeze)
    scatter!(ax, abs2(A_stop), abs2(Array(first.(steady_state))[N÷4, end]), color=:black, markersize=16)
    fig
end
##
function batch_solve(steady_state, lengths, batchsize, nbatches, tspan, δt, path; param, kwargs...)
    u0_steady = stack(steady_state for _ ∈ 1:batchsize)
    noise_prototype = similar(u0_steady, real(eltype(u0_steady)))

    prob = GrossPitaevskiiProblem(u0_steady, lengths; noise_prototype, param, kwargs...)
    solver = StrangSplittingB(1, δt)

    h5open(path, "cw") do file
        @showprogress for batch ∈ 1:nbatches.+ length(file)
            sol = solve(prob, solver, tspan; save_start=false, show_progress=false)[2]
            file["batch_$batch"] = Array(dropdims(sol, dims=3))
            HDF5.attributes(file["batch_$batch"])["lengths"] = [x for x in lengths]
        end
    end
end

tspan_noise = (0.0f0, 50.0f0) .+ tspan_steady[end]
path = "/home/stagios/Marcos/LEON_Marcos/Users/Marcos/hawking_posp.h5"

batch_solve(steady_state, lengths, 10^5, 10, tspan_noise, δt, path;
    dispersion, potential, nonlinearity, pump, param, noise_func)
##

file = h5open(path)
sol = read(first(file))

reinterpret(reshape, ComplexF32, sol)

[1]
