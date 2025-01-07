using GeneralizedGrossPitaevskii, CairoMakie, LinearAlgebra, CUDA, Statistics

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
    if x[1] ≤ -param.L * 0.9 / 2 || x[1] ≥ -2
        a *= 0
    elseif -param.L * 0.9 / 2 < x[1] ≤ -param.L * 0.85 / 2
        a *= 4
    end

    a * cis(mapreduce(*, +, param.k_pump, x))
end

noise_func(ψ, param) = √(param.γ / 2 / param.δL)

# Space parameters
L = 250.0f0
lengths = (L,)
N = 256
δL = L / N
rs = range(; start=-L / 2, step=L / N, length=N)

# Polariton parameters
ħ = 0.6582f0 #meV.ps
γ = 0.047f0 / ħ
m = ħ^2 / (2 * 1.29f0)
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
Imax = 90.0f0
Amax = √Imax
t_cycle = 250.0f0
t_freeze = 236.0f0

# Full parameter tuple
param = (; δ₀, m, γ, ħ, L, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, t_freeze, δL, k_pump)

function reduction(ψ, param)
    # mapreduce(x -> abs2.(x), +, eachslice(ψ, dims=2)) / size(ψ)[end]

    steady = mean(ψ, dims=2)
    δψ = reshape(ψ .- steady, 1, size(ψ)...)
    δψ′ = permutedims(δψ, (2, 1, 3))
    n = dropdims(mean(abs2, δψ, dims=3), dims=(1, 3)) .- 1 / 2param.δL

    corr = mapreduce((x, y) -> abs2.(x) .* abs2.(y), +, eachslice(δψ, dims=3), eachslice(δψ′, dims=3)) / size(ψ)[end]
    δ = one(corr)
    (corr - (1 .+ δ) .* (n .+ n' .+ 1 / 2param.δL) / 2param.δL) ./ (n .* n')
end

u0 = CUDA.randn(ComplexF32, N, 5 * 10^4) / √(2δL)
noise_prototype = similar(u0)
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, potential, nonlinearity, pump, param, noise_func, noise_prototype)
tspan = (0, 400.0f0)
solver = StrangSplittingB(1, 5.0f-2)
ts, sol = solve(prob, solver, tspan; save_start=false, reduction)

J = 256-50:256+50
lines(rs[:], real(Array(sol)[:, 1]))
##
J = N÷2-50:N÷2+50
heatmap(rs[J], rs[J], real(Array(sol)[J, J, 1]), colorrange=(0, 7))