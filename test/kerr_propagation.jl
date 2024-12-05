function test_kerr_propagation(ψ₀, xs, ys, δt, nsteps, g)
    lengths = @. -2 * first((xs, ys))
    ts = range(; start=0, step=δt, length=nsteps + 1)
    function dispersion!(dest, ks...; param=nothing)
        dest[1] = sum(abs2, ks) / 2
    end
    prob = GrossPitaevskiiProblem(dispersion!, nothing, g, nothing, ψ₀, lengths)
    ψs = dropdims(solve(prob, StrangSplitting(), nsteps, nsteps, δt); dims=1)
    @test ψs ≈ kerr_propagation(dropdims(ψ₀, dims=1), xs, ys, ts, nsteps, g=2g[1])
end

@testset "Scalar Kerr Propagation" begin
    for n ∈ 1:5
        N = 256
        L = 10.0f0
        ΔL = L / N
        δt = 0.04f0 * rand(Float32)
        nsteps = rand(50:200)
        g = 4.0f0 * rand(Float32, 1, 1)

        rs = range(; start=-L / 2, length=N, step=ΔL)
        ψ₀ = reshape(lg(rs, rs, l=rand(-5:5)) + lg(rs, rs, l=rand(-5:5)), 1, N, N)

        test_kerr_propagation(ψ₀, rs, rs, δt, nsteps, g)
    end
end