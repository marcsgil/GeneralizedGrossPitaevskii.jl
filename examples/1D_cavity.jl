using GeneralizedGrossPitaevskii, CairoMakie, LinearAlgebra, CUDA, Statistics, ProgressMeter, KernelAbstractions

function dispersion(ks, param)
    -im * param.γ / 2 + param.ħ * sum(abs2, ks) / 2param.m - param.δ₀
end

function potential(rs, param)
    param.V_def * exp(-sum(abs2, rs) / param.w_def^2) +
    param.V_damp * damping_potential(rs, -param.L / 2, param.L / 2, param.w_damp)
end

function A(t, Amax, t_cycle, t_freeze)
    _t = ifelse(t > t_freeze, t_freeze, t)
    val = Amax * _t * (t_cycle - _t) * 4 / t_cycle^2
    val < 0 ? zero(val) : val
end

function pump(x, param, t)
    a = A(t, param.Amax, param.t_cycle, param.t_freeze)
    if x[1] ≤ -param.L * 0.9 / 2 || x[1] ≥ -7
        a *= 0
    elseif -param.L * 0.9 / 2 < x[1] ≤ -param.L * 0.85 / 2
        a *= 1
    end
    a * cis(mapreduce(*, +, param.k_pump, x))
end

noise_func(ψ, param) = √(param.γ / 2 / param.δL)

# Space parameters
L = 1024.0f0
lengths = (L,)
N = 1024
δL = L / N
rs = range(; start=-L / 2, step=L / N, length=N)

# Polariton parameters
ħ = 0.6582f0 #meV.ps
γ = 0.047f0 / ħ
#m = ħ^2 / (2 * 1.29f0)
m = 1f0
nonlinearity = 0.0003f0 / ħ
δ₀ = 0.49 / ħ

# Potential parameters
V_damp = 10.0f0
w_damp = 1.0f0
V_def = -0.85f0 / ħ
w_def = 0.75f0

# Pump parameters
k_pump = 0.25f0
δ = δ₀ - ħ * k_pump^2 / 2m

# Bistability cycle parameters
Imax = 130.0f0
Amax = √Imax
t_cycle = 300.0f0
t_freeze = 275.0f0

δt = 1.0f-1

# Full parameter tuple
param = (; δ₀, m, γ, ħ, L, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, t_freeze, δL, k_pump)

u0_empty = CUDA.zeros(ComplexF32, N)
prob_steady = GrossPitaevskiiProblem(u0_empty, lengths; dispersion, potential, nonlinearity, pump, param)
tspan_steady = (0, 800.0f0)
solver_steady = StrangSplittingB(512, δt)
ts_steady, sol_steady = solve(prob_steady, solver_steady, tspan_steady)

steady_state = @view sol_steady[:, end]

heatmap(rs, ts_steady, Array(abs2.(sol_steady)))
##
J = N÷2-70:N÷2+70
lines(rs[J], Array(abs2.(steady_state[J])))

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
    solver = StrangSplittingB(1, δt)

    one_point = similar(steady_state, real(eltype(steady_state)))
    two_point = similar(steady_state, real(eltype(steady_state)), size(steady_state, 1), size(steady_state, 1))

    fill!(one_point, 0)
    fill!(two_point, 0)

    for batch ∈ 1:nbatches
        @info "Batch $batch"
        ts, _sol = solve(prob, solver, tspan; save_start=false, reuse_u0=true)
        sol = dropdims(_sol, dims=3)

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

G2 = calculate_correlation(steady_state, lengths, 10^5, 10^2, tspan_noise, δt;
    dispersion, potential, nonlinearity, pump, param, noise_func)

##
J = N÷2-70:N÷2+70

with_theme(theme_latexfonts()) do
    fig = Figure(; size=(730, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel = L"x", ylabel = L"x\prime")
    hm = heatmap!(ax, rs[J], rs[J], (Array(real(G2)[J, J]) .- 1) * 1e4, colorrange=(-1, 1))
    Colorbar(fig[1,2], hm, label = L"g_2(x, x\prime) -1 \ \ ( \times 10^{-4})")
    save("dev_env/g2m1.pdf", fig)
    fig
end