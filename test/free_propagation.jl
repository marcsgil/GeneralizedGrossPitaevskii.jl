@testset "Scalar Free Propagation" begin
    for n ∈ 1:5
        N = 64
        L = rand(4.0f0:1.0f0:10.0f0)
        lengths = (L, L)
        ΔL = L / N
        dt = 0.3f0 * rand(Float32)
        nsaves = rand(50:200)

        rs = range(; start=-L / 2, length=N, step=ΔL)
        u0 = (lg(rs, rs, l=rand(1:5)) + lg(rs, rs, l=rand(1:5)),)

        dispersion(ks, param) = sum(abs2, ks) / 2
        prob = GrossPitaevskiiProblem(u0, lengths; dispersion)

        tspan = (0, nsaves * dt)
        for alg ∈ (StrangSplittingA(), StrangSplittingB(), StrangSplittingC(), SimpleAlg())
            ts, sol = solve(prob, alg, tspan; nsaves, dt, show_progress=false)
            @test sol[1] ≈ free_propagation(u0[1], rs, rs, ts)

            for type ∈ (identity, SVector, SMatrix{1,1})
                if alg isa SimpleAlg
                    type != identity && continue
                end
                new_dispersion(args...) = type(dispersion(args...))
                prob2 = GrossPitaevskiiProblem(u0, lengths; dispersion=new_dispersion)
                ts, sol = solve(prob2, alg, tspan; nsaves, dt, show_progress=false)
                @test sol[1] ≈ free_propagation(u0[1], rs, rs, ts)
            end
        end
    end
end