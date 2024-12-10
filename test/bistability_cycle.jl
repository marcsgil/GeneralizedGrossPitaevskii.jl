using GeneralizedGrossPitaevskii

@testset "Bistability Cycle" begin
    ω₀ = 1483.0f0
    g = 0.01f0
    δ = 0.3f0
    ωₚ = ω₀ + δ
    kz = 27.0f0
    γ = 0.1f0

    function dispersion(ks, param)
        tmax, Imax, width, ωₚ, ω₀, kz, γ = param
        -im * γ / 2 + ω₀ * (1 + sum(abs2, ks) / 2kz^2) - ωₚ
    end

    potential = nothing

    function I(t, tmax, Imax)
        val = -Imax * t * (t - tmax) * 4 / tmax^2
        val < 0 ? zero(val) : val
    end

    function pump(x, param, t)
        tmax, Imax, width = param
        exp(-sum(abs2, x) / width^2) * √I(t, tmax, Imax)
    end

    L = 256.0f0
    lengths = (L,)
    u0 = zeros(ComplexF32, ntuple(n -> 256, length(lengths)))

    Imax = 0.6f0
    width = 50.0f0

    δt = 0.1f0
    nsaves = 512
    tspan = (0, 3300)
    tmax = tspan[end]

    param = (tmax, Imax, width, ωₚ, ω₀, kz, γ)

    prob = GrossPitaevskiiProblem(u0, lengths, dispersion, potential, g, pump, param)
    solver = StrangSplitting(nsaves, δt)

    ts, sol = solve(prob, solver, tspan; show_progress=false)

    function bistability_curve(n, δ, g, γ)
        n * (γ^2 / 4 + (g * n - δ)^2)
    end
    Is = I.(ts, tmax, Imax)

    error = similar(Is)

    for (n, slice) ∈ enumerate(eachslice(sol, dims=ndims(sol)))
        Is_pred = bistability_curve(maximum(abs2, slice), δ, g, γ)
        error[n] = abs(Is_pred - Is[n])
    end

    @test sum(error[140:400]) / length(Is) ≤ 3e-3
end