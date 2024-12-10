function test_free_propagation(u0, xs, ys, δt, nsteps)
    lengths = @. -2 * first((xs, ys))
    prob = free_propagation_template(u0, lengths)
    solver = StrangSplitting(nsteps, δt)
    tspan = (0, nsteps * δt)
    ts, sol = solve(prob, solver, tspan; show_progress=false)
    @test sol ≈ free_propagation(u0, xs, ys, ts)
    vector_prob = scalar2vector(prob)
    ts, vector_sol = solve(vector_prob, solver, tspan; show_progress=false)
    @test sol ≈ dropdims(vector_sol, dims=1)
end

@testset "Scalar Free Propagation" begin
    for n ∈ 1:5
        N = 64
        L = rand(4.0f0:1.0f0:10.0f0)
        ΔL = L / N
        δt = 0.3f0 * rand(Float32)
        nsteps = rand(50:200)

        rs = range(; start=-L / 2, length=N, step=ΔL)
        ψ₀ = lg(rs, rs, l=rand(1:5)) + lg(rs, rs, l=rand(1:5))

        test_free_propagation(ψ₀, rs, rs, δt, nsteps)
    end
end