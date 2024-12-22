function test_kerr_propagation(u0, xs, ys, δt, nsteps, g)
    lengths = @. -2 * first((xs, ys))
    prob = kerr_propagation_template(u0, lengths, g)
    solver = StrangSplittingA(nsteps, δt)
    tspan = (0, nsteps * δt)

    for Tsolver ∈ (StrangSplittingA, StrangSplittingB, StrangSplittingC)
        solver = Tsolver(nsteps, δt)
        ts, sol = solve(prob, solver, tspan; show_progress=false)
        @test sol ≈ kerr_propagation(u0, xs, ys, ts, nsteps; g=-g)
        vector_prob = scalar2vector(prob)
        ts, vector_sol = solve(vector_prob, solver, tspan; show_progress=false)
        @test sol ≈ dropdims(vector_sol, dims=1)
    end
end

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
        prob = GrossPitaevskiiProblem(u0, lengths; dispersion=ScalarFunction(dispersion), nonlinearity=g/2)

        tspan = (0, nsteps * δt)
        for Tsolver ∈ (StrangSplittingA, StrangSplittingB, StrangSplittingC)
            solver = Tsolver(nsteps, δt)
            ts, sol = solve(prob, solver, tspan; show_progress=false)
            @test sol ≈ kerr_propagation(u0, rs, rs, ts, nsteps; g=-g)
            
            new_u0 = reshape(u0, 1, size(u0)...)
            for type ∈ (VectorFunction, MatrixFunction), nonlinearity ∈ (g/2, [g/2], [g/2;;])
                new_dispersion = type((dest, args...) -> dest[1] = dispersion(args...))
                prob2 = GrossPitaevskiiProblem(new_u0, lengths; dispersion=new_dispersion, nonlin'earity)
                ts, sol = solve(prob2, solver, tspan; show_progress=false)
                @test dropdims(sol, dims=1) ≈ kerr_propagation(u0, rs, rs, ts, nsteps; g=-g)
            end
        end
    end
end