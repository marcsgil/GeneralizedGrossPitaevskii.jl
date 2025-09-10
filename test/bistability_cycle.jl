@testset "Bistability Cycle" begin
    ω₀ = 1483
    g = 0.01
    δ = 0.3
    ωₚ = ω₀ + δ
    kz = 27
    γ = 0.1

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

    dispersion2(ks, param) = SVector(dispersion(ks, param))
    dispersion3(ks, param) = SMatrix{1,1}(dispersion(ks, param))

    nonlinearity2(ψ, param) = SVector(nonlinearity(ψ, param))
    nonlinearity3(ψ, param) = SMatrix{1,1}(nonlinearity(ψ, param))

    pump2(x, param, t) = SVector(pump(x, param, t))
    pump3(x, param, t) = SMatrix{1,1}(pump(x, param, t))

    L = 256
    lengths = (L,)
    u0 = (zeros(ComplexF64, ntuple(n -> 256, length(lengths))),)

    Imax = 0.6
    width = 50

    dt = 0.05
    nsaves = 512
    tspan = (0, 3300)
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

    for dispersion ∈ (dispersion, dispersion2, dispersion3), nonlinearity ∈ (nonlinearity, nonlinearity2, nonlinearity3), pump ∈ (pump, pump2, pump3)
        prob2 = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity, pump, param)
        ts, sol2 = solve(prob2, alg, tspan; nsaves, dt, show_progress=false)
        @test sol2[1] ≈ sol[1]
    end
end