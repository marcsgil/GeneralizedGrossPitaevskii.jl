function test_free_propagation(u₀, xs, ys, δt, nsteps)
    lengths = @. -2 * first((xs, ys))
    ts = range(; start=0, step=δt, length=nsteps + 1)
    function dispersion!(dest, ks...; param=nothing)
        dest[1] = sum(abs2, ks) / 2
    end
    prob = GrossPitaevskiiProblem(dispersion!, nothing, nothing, nothing, u₀, lengths)
    ψs = dropdims(solve(prob, StrangSplitting(), nsteps, nsteps, δt); dims=1)
    @test ψs ≈ free_propagation(dropdims(u₀, dims=1), xs, ys, ts)
end

@testset "Scalar Free Propagation" begin
    for n ∈ 1:5
        N = 256
        L = rand(4.0f0:1.0f0:10.0f0)
        ΔL = L / N
        δt = 0.3f0 * rand(Float32)
        nsteps = rand(50:200)

        rs = range(; start=-L / 2, length=N, step=ΔL)
        ψ₀ = reshape(lg(rs, rs, l=rand(1:5)) + lg(rs, rs, l=rand(1:5)), 1, N, N)

        test_free_propagation(ψ₀, rs, rs, δt, nsteps)
    end
end