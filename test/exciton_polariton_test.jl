@testset "Exciton Polariton Test" begin
    function dispersion(k, param)
        Dcc = param.ħ * sum(abs2, k) / 2param.m - param.δc - im * param.γc
        Dxx = -param.δx - im * param.γx
        Dxc = param.Ωr
        @SMatrix [Dcc Dxc; Dxc Dxx]
    end

    nonlinearity(ψ, param) = @SVector [0, param.g * abs2(ψ[2])]

    function pump(r, param, t)
        SVector(param.A * exp(-sum(abs2, r .- param.L / 2) / param.w^2), 0f0)
    end

    ħ = 0.654f0 # (meV*ps)
    Ωr = 5.07f0 / 2ħ
    γx = 0.0015f0 / ħ
    γc = 0.07f0 / 0.6571f0 / ħ
    ωx = 1484.44f0 / ħ
    ωc = 1482.76f0 / ħ
    m = ħ^2 / (2 * 2.0f-1)

    ωp = ωc
    δx = ωp - ωx
    δc = ωp - ωc

    A = 2f0
    w = 100f0

    g = 1f-2 / ħ

    L = 256f0
    N = 128
    lengths = (L, L)
    u0 = (zeros(ComplexF32, N, N), zeros(ComplexF32, N, N))

    param = (; ħ, m, ωc, δc, γc, δx, γx, Ωr, A, w, g, L)
    prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity, pump, param)

    nsaves = 256
    dt = 1f-1
    tspan = (0f0, 100f0)

    for alg ∈ (StrangSplitting(),)
        ts, sol = solve(prob, alg, tspan; nsaves, dt, show_progress=false)

        nx = abs2.(last(sol))[N÷2, N÷2, end]
        nc = abs2.(first(sol))[N÷2, N÷2, end]

        @test abs(abs2(Ωr - (δx + im * γx / 2 - g * nx) * (δc + im * γc / 2) / Ωr) * nx / abs2(A) - 1) < 3e-2

        @test abs(abs2(δx + im * γx / 2 - g * nx) * nx / Ωr^2 / nc - 1) < 3e-2
    end
end