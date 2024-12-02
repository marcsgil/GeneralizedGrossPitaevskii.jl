function test_free_propagation(ψ₀, xs, ys, δt, nsteps)
    lengths = @. -2 * first((xs, ys))
    ts = range(; start=0, step=δt, length=nsteps + 1)
    A(kx, ky) = -im * (kx^2 + ky^2) / 2
    prob = GrossPitaevskiiProblem(ψ₀, A, nothing, nothing, nothing, δt, lengths)
    ψs = dropdims(solve(prob, nsteps, 1); dims=1)
    @test ψs ≈ free_propagation(ψ₀, xs, ys, ts)
end

@testset "Scalar Free Propagation" begin
    for n ∈ 1:5
        N = 256
        L = rand(4.0f0:1.0f0:10.0f0)
        ΔL = L / N
        δt = 0.3f0 * rand(Float32)
        nsteps = rand(50:200)

        rs = range(; start=-L / 2, length=N, step=ΔL)
        ψ₀ = lg(rs, rs, l=rand(1:5)) + lg(rs, rs, l=rand(1:5))

        test_free_propagation(ψ₀, rs, rs, δt, nsteps)
    end
end