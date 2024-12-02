function test_kerr_propagation(ψ₀, xs, ys, δt, nsteps, g)
    lengths = @. -2 * first((xs, ys))
    ts = range(; start=0, step=δt, length=nsteps + 1)
    A(kx, ky) = -im * (kx^2 + ky^2) / 2
    prob = GrossPitaevskiiProblem(ψ₀, A, nothing, nothing, g, δt, lengths)
    ψs = dropdims(solve(prob, nsteps, 1); dims=1)
    @test ψs ≈ kerr_propagation(ψ₀, xs, ys, ts, nsteps, g=2g)
end

@testset "Scalar Kerr Propagation" begin
    for n ∈ 1:5
        N = 256
        L = 10.0f0
        ΔL = L / N
        δt = 0.05f0 * rand(Float32)
        nsteps = rand(50:200)
        g = 4.0f0 * rand(Float32)

        rs = range(; start=-L / 2, length=N, step=ΔL)
        ψ₀ = lg(rs, rs, l=rand(-5:5)) + lg(rs, rs, l=rand(-5:5))

        test_kerr_propagation(ψ₀, rs, rs, δt, nsteps, g)
    end
end