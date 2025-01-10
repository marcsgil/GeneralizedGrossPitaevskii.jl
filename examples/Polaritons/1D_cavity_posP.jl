using GeneralizedGrossPitaevskii, CairoMakie, LinearAlgebra, CUDA, Statistics, ProgressMeter, KernelAbstractions, FFTW, Revise
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
    val = √(im * param.g)
    SVector(val * ψ[1], conj(val) * ψ[2])
end

function nonlinearity(ψ, param)
    val = param.g * prod(ψ)
    SVector(val, -val)
end

# Space parameters
L = 400.0f0
lengths = (L,)
N = 1024
δL = L / N
rs = range(; start=-L / 2, step=L / N, length=N)

# Polariton parameters
ħ = 0.6582f0 #meV.ps
γ = 0.047f0 / ħ
m = ħ^2 / 2.5f0
g = 0.0003f0 / ħ
δ₀ = 0.49 / ħ

# Potential parameters
V_damp = 100.0f0
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

δt = 2.0f-2

# Full parameter tuple
param = (; δ₀, m, γ, ħ, L, g, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, t_freeze, δL, k_pump)

u0_empty = CUDA.fill(SVector{2,ComplexF32}(0, 0), N, 100)
prob_steady = GrossPitaevskiiProblem(u0_empty, lengths; dispersion, potential, nonlinearity, pump, param)
tspan_steady = (0, 800.0f0)
solver_steady = StrangSplittingB(512, δt)
ts_steady, sol_steady = solve(prob_steady, solver_steady, tspan_steady);
##

y = CUDA.randn(ComplexF32, 2, 1024, 10^5)
x = similar(y)

plan = plan_fft(x, 1)

@benchmark mul!($y, $plan, $x)

perm = (2, 3, 1)

z = permutedims(y, perm)

@benchmark permutedims!($z, $y, $perm)

permutedims!

##
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
function one_point_corr!(dest, sol)
    backend = get_backend(dest)

    @kernel function kernel!(dest, sol)
        j = @index(Global)
        x = 0f0
        for k ∈ axes(sol, 2)
            x += real(prod(sol[j, k]))
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
        x = 0f0
        for m ∈ axes(sol, 2)
            x += real(prod(sol[j, m]) * prod(sol[k, m]))
        end
        dest[j, k] += x
    end

    kernel!(backend, 64)(dest, sol, ndrange=size(dest))
    KernelAbstractions.synchronize(backend)
end

function calculate_correlation(steady_state, lengths, batchsize, nbatches, tspan, δt; param, kwargs...)
    u0_steady = stack(steady_state for _ ∈ 1:batchsize)
    #u0_steady = CUDA.randn(eltype(steady_state), length(steady_state), batchsize) ./ 2param.δL .+ steady_state
    noise_prototype = similar(u0_steady, real(eltype(u0_steady)))

    prob = GrossPitaevskiiProblem(u0_steady, lengths; param, kwargs...)
    solver = StrangSplittingB(1, δt)

    one_point = similar(steady_state, real(eltype(eltype(steady_state))))
    two_point = similar(steady_state, real(eltype(eltype(steady_state))), size(steady_state, 1), size(steady_state, 1))

    fill!(one_point, 0f0)
    fill!(two_point, 0f0)

    for batch ∈ 1:nbatches
        @info "Batch $batch"
        ts, _sol = solve(prob, solver, tspan; save_start=false)
        sol = dropdims(_sol, dims=3)

        one_point_corr!(one_point, sol)
        two_point_corr!(two_point, sol)
    end

    one_point /= nbatches * batchsize
    two_point /= nbatches * batchsize

    two_point ./ (one_point .* one_point')
end

tspan_noise = (0.0f0, 50.0f0) .+ tspan_steady[end]

G2 = calculate_correlation(steady_state, lengths, 10^4, 1, tspan_noise, δt;
    dispersion, potential, nonlinearity, pump, param, noise_func)
##
J = N÷2-260:N÷2+260

with_theme(theme_latexfonts()) do
    fig = Figure(; size=(730, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"x", ylabel=L"x\prime")
    hm = heatmap!(ax, rs[J], rs[J], (Array(real(G2)[J, J]) .- 1) * 1e5, colorrange=(-5, 5), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(x, x\prime) -1 \ \ ( \times 10^{-4})")
    #save("dev_env/g2m1.pdf", fig)
    fig
end