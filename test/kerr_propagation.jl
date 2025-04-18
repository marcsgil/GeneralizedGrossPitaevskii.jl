@testset "Scalar Kerr Propagation" begin
    for n ∈ 1:5
        N = 64
        L = 10.0f0
        lengths = (L, L)
        ΔL = L / N
        dt = 0.01f0 * rand(Float32)
        nsaves = rand(50:200)
        g = 1.0f0 * randn(Float32)

        rs = range(; start=-L / 2, length=N, step=ΔL)
        u0 = (lg(rs, rs, l=rand(-5:5)) + lg(rs, rs, l=rand(-5:5)),)

        dispersion(ks, param) = sum(abs2, ks) / 2
        nonlinearity(ψ, param) = param.g * abs2.(ψ) / 2

        param = (; g)
        prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity, param)

        tspan = (0, nsaves * dt)
        alg = StrangSplitting()

        ts, sol = solve(prob, alg, tspan; nsaves, dt, show_progress=false)
        good_sol = kerr_propagation(u0[1], rs, rs, ts, nsaves; g=-g)
        @test sol[1] ≈ good_sol

        for type ∈ (identity, SVector, SMatrix{1,1}), type′ ∈ (identity, SVector, SMatrix{1,1})
            new_dispersion(args...) = type(dispersion(args...))
            new_nonlinearity(args...) = type(nonlinearity(args...))
            prob2 = GrossPitaevskiiProblem(u0, lengths; dispersion=new_dispersion, nonlinearity=new_nonlinearity, param)
            ts, sol = solve(prob2, alg, tspan; nsaves, dt, show_progress=false)
            @test sol[1] ≈ good_sol
        end
    end
end