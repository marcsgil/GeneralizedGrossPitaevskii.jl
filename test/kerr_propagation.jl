function test_kerr_propagation(u0, xs, ys, δt, nsteps, g)
    lengths = @. -2 * first((xs, ys))
    prob = kerr_propagation_template(u0, lengths, g)
    solver = StrangSplitting(nsteps, δt)
    tspan = (0, nsteps * δt)
    ts, sol = solve(prob, solver, tspan; show_progress=false)
    @test sol ≈ kerr_propagation(u0, xs, ys, ts, nsteps; g=-g)
    vector_prob = scalar2vector(prob)
    ts, vector_sol = solve(vector_prob, solver, tspan; show_progress=false)
    @test sol ≈ dropdims(vector_sol, dims=1)
end

@testset "Scalar Kerr Propagation" begin
    for n ∈ 1:5
        N = 64
        L = 10.0f0
        ΔL = L / N
        δt = 0.01f0 * rand(Float32)
        nsteps = rand(50:200)
        g = 1.0f0 * randn(Float32)

        rs = range(; start=-L / 2, length=N, step=ΔL)
        ψ₀ = lg(rs, rs, l=rand(-5:5)) + lg(rs, rs, l=rand(-5:5))

        test_kerr_propagation(ψ₀, rs, rs, δt, nsteps, g)
    end
end