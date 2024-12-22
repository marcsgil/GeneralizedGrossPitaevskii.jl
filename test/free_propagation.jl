@testset "Scalar Free Propagation" begin
    for n ∈ 1:5
        N = 64
        L = rand(4.0f0:1.0f0:10.0f0)
        lengths = (L, L)
        ΔL = L / N
        δt = 0.3f0 * rand(Float32)
        nsteps = rand(50:200)

        rs = range(; start=-L / 2, length=N, step=ΔL)
        u0 = lg(rs, rs, l=rand(1:5)) + lg(rs, rs, l=rand(1:5))

        dispersion(ks, param) = sum(abs2, ks) / 2
        prob = GrossPitaevskiiProblem(u0, lengths; dispersion=ScalarFunction(dispersion))

        tspan = (0, nsteps * δt)
        for Tsolver ∈ (StrangSplittingA, StrangSplittingB, StrangSplittingC)
            solver = Tsolver(nsteps, δt)
            ts, sol = solve(prob, solver, tspan; show_progress=false)
            @test sol ≈ free_propagation(u0, rs, rs, ts)

            new_u0 = reshape(u0, 1, size(u0)...)
            for type ∈ (VectorFunction, MatrixFunction)
                new_dispersion = type((dest, args...) -> dest[1] = dispersion(args...))
                prob2 = GrossPitaevskiiProblem(new_u0, lengths; dispersion=new_dispersion)
                ts, sol = solve(prob2, solver, tspan; show_progress=false)
                @test dropdims(sol, dims=1) ≈ free_propagation(u0, rs, rs, ts)
            end
        end
    end
end