using GeneralizedGrossPitaevskii, CairoMakie, LinearAlgebra, CUDA, Statistics, StructuredLight, FFTW

function dispersion(ks, param)
    -im * param.γ / 2 + sum(abs2, ks) / 2param.kz^2 - param.δ₀
end

function pump(rs, param, t)
    param.A * exp(-sum(abs2, rs) / param.w^2) * cis(mapreduce(*, +, param.kₚ, rs)) * (1-exp(-t/50))
end

noise_func(ψ, param) = √(param.γ / 2 / param.δV)

# Space parameters
L = 256.0f0
lengths = (L, L)
N = 128
δV = prod(lengths ./ N)
rs = range(; start=-L / 2, step=L / N, length=N)

# Polariton parameters
ħ = 0.6582f0 #meV.ps
γ = 0.047f0 / ħ
kz = 27f0
nonlinearity = 0.0003f0 / ħ
δ₀ = 0.49 / ħ

# Pump parameters
kₚ = (1f0, 1f0)
A = 11f0
w = 50f0

param = (; δV, γ, kz, δ₀, kₚ, A, w)

u0 = CUDA.zeros(ComplexF32, N, N, 1000)
noise_prototype = similar(u0)
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity, pump, param, noise_func, noise_prototype)

solver = StrangSplittingB(1, 1f0)
tspan = (0f0, 400f0)

ts, sol = solve(prob, solver, tspan; save_start=false)
##
save_animation(Array(abs2.(sol)), "dev_env/test.mp4", share_colorrange=true)
##
lines(Array(abs2.(sol[N÷2,N÷2,:])))
##

sol
sol_k = fftshift(fft(ifftshift(sol, (1,2)), (1,2)), (1,2))

Ik = dropdims(mean(abs2, sol_k, dims=3), dims=(3,4))

heatmap(log.(real(Array(Ik))))