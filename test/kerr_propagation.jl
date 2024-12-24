@testset "Scalar Kerr Propagation" begin
    for n ∈ 1:5
        N = 64
        L = 10.0f0
        lengths = (L, L)
        ΔL = L / N
        δt = 0.01f0 * rand(Float32)
        nsteps = rand(50:200)
        g = 1.0f0 * randn(Float32)

        rs = range(; start=-L / 2, length=N, step=ΔL)
        u0 = lg(rs, rs, l=rand(-5:5)) + lg(rs, rs, l=rand(-5:5))

        dispersion(ks, param) = sum(abs2, ks) / 2
        prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity=g / 2)

        tspan = (0, nsteps * δt)
        for Tsolver ∈ (StrangSplittingA, StrangSplittingB, StrangSplittingC)
            solver = Tsolver(nsteps, δt)
            ts, sol = solve(prob, solver, tspan; show_progress=false)
            good_sol = kerr_propagation(u0, rs, rs, ts, nsteps; g=-g)
            @test sol ≈ good_sol

            new_u0 = [SVector(val) for val ∈ u0]
            for type ∈ (identity, SVector, SMatrix{1,1}), nonlinearity ∈ (g / 2, SVector(g / 2), SMatrix{1,1}(g / 2))
                new_dispersion(args...) = type(dispersion(args...))
                prob2 = GrossPitaevskiiProblem(new_u0, lengths; dispersion=new_dispersion, nonlinearity)
                ts, sol = solve(prob2, solver, tspan; show_progress=false)
                @test reinterpret(reshape, ComplexF32, sol) ≈ good_sol
            end
        end
    end
end