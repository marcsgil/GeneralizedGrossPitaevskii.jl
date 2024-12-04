using StructuredLight, CairoMakie, GeneralizedGrossPitaevskii, ProgressMeter, FFTW, LinearAlgebra

function bistability_curve(n, δ, g, γ)
    n * (γ^2 / 4 + (g * n - δ)^2)
end
##
N = 256
L = 256.0f0
lengths = (L, L)
ΔL = L / N
δt = 0.01f0
nsteps = 100
g = 4.0f0

rs = range(; start=-L / 2, length=N, step=ΔL)

ψ₀ = zeros(ComplexF32, N, N)
F(w, rs...) = exp(-sum(abs2, rs) / w^2)

ω₀ = 1483.0f0
δ = 0.3f0
ωₚ = ω₀ + δ
kz = 27.0f0
γ = 0.1f0

A(γ, ωₚ, ω₀, kz, ks...) = -γ / 2 + im * (ωₚ - ω₀ * (1 + sum(abs2, ks) / 2kz^2))
A(ks...) = A(γ, ωₚ, ω₀, kz, ks...)

g = 1.0f-2
δt = 1.0f-1
nsteps = 2^11
save_every = 2^3

ns_theo = LinRange(0, 43, 512)
Is_theo = bistability_curve.(ns_theo, δ, g, γ)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=16)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(ax, Is_theo, ns_theo, color=:blue, linewidth=4, label="Theoretical")

    fig
end
##
progress = Progress(nsteps)
prob = GrossPitaevskiiProblem(ψ₀, A, nothing, (rs...) -> F(50, rs...), -g, δt, lengths)

sol = dropdims(solve(prob, nsteps, save_every; progress), dims=1)
finish!(progress)

save_animation(abs2.(sol), "examples/test.mp4", scaling=1, share_colorrange=true)

n = maximum(abs2, sol[:, :, end])
bistability_curve(n, δ, g, γ)
##
Is_sim = LinRange(extrema(Is_theo)..., 8)
Is_sim = vcat(Is_sim, reverse(Is_sim))
ns_sim = similar(Is_sim)

drive = copy(prob.drive)
fill!(prob.ψ, 0)

progress = Progress(length(ns_sim) * nsteps)
for j ∈ eachindex(ns_sim, Is_sim)
    sol = dropdims(solve(prob, nsteps, save_every; progress), dims=1)
    ifftshift!(view(prob.ψ, 1, :, :), sol[:, :, end])
    @. prob.drive = drive * √Is_sim[j]
    ns_sim[j] = maximum(abs2, sol[:, :, end])
end
finish!(progress)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=16)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")

    first_Is = Is_sim[begin:length(Is_sim)÷2]
    first_ns = ns_sim[begin:length(ns_sim)÷2]
    second_Is = Is_sim[length(Is_sim)÷2+1:end]
    second_ns = ns_sim[length(ns_sim)÷2+1:end]
    lines!(ax, Is_theo, ns_theo, color=:blue, linewidth=4, label="Theoretical")
    scatter!(ax, first_Is, first_ns, color=:red, markersize=12, label="Simulation (Increasing I)")
    scatter!(ax, second_Is, second_ns, color=:black, marker=:cross, markersize=12, label="Simulation (Decreasing I)")

    axislegend(ax, position=(1, 0.3))

    #save("Plots/bistability.png", fig)

    fig
end