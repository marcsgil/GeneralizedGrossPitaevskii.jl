@testset "Bistability Cycle" begin
    ω₀ = 1483.0f0
    g = 0.01f0
    δ = 0.3f0
    ωₚ = ω₀ + δ
    kz = 27.0f0
    γ = 0.1f0

    function dispersion(ks, param)
        -im * param.γ / 2 + param.ω₀ * (1 + sum(abs2, ks) / 2param.kz^2) - param.ωₚ
    end

    nonlinearity(ψ, param) = param.g * abs2.(ψ)

    function I(t, tmax, Imax)
        val = -Imax * t * (t - tmax) * 4 / tmax^2
        val < 0 ? zero(val) : val
    end

    function pump(x, param, t)
        exp(-sum(abs2, x .- param.L / 2) / param.width^2) * √I(t, param.tmax, param.Imax)
    end

    function bistability_curve(n, δ, g, γ)
        n * (γ^2 / 4 + (g * n - δ)^2)
    end

    L = 256.0f0
    lengths = (L,)
    u0 = (zeros(ComplexF32, ntuple(n -> 256, length(lengths))),)

    Imax = 0.6f0
    width = 50.0f0

    dt = 0.05f0
    nsaves = 512
    tspan = (0, 3300f0)
    tmax = tspan[end]

    param = (; tmax, Imax, width, ωₚ, ω₀, kz, γ, g, L)

    prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity, pump, param)
    alg = StrangSplitting()

    ts, sol = solve(prob, alg, tspan; nsaves, dt, show_progress=false)

    Is = I.(ts, tmax, Imax)

    error = similar(Is)

    for (n, slice) ∈ enumerate(eachslice(sol[1], dims=ndims(sol[1])))
        Is_pred = bistability_curve(maximum(abs2, slice), δ, g, γ)
        error[n] = abs(Is_pred - Is[n])
    end

    @test sum(error[140:400]) / length(Is) ≤ 3e-3

    for type ∈ (identity, SVector, SMatrix{1,1}), type′ ∈ (identity, SVector, SMatrix{1,1}), pump_type ∈ (identity, SVector)
        new_dispersion(args...) = type(dispersion(args...))
        new_nonlinearity(args...) = type′(nonlinearity(args...))
        new_pump(args...) = pump_type(pump(args...))
        prob2 = GrossPitaevskiiProblem(u0, lengths; dispersion=new_dispersion, nonlinearity=new_nonlinearity, pump, param)
        ts, sol2 = solve(prob2, alg, tspan; nsaves, dt, show_progress=false)
        @test sol2[1] ≈ sol[1]
    end
end