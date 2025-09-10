@testset "Scalar Free Propagation" begin
    for n ∈ 1:5
        N = 64
        L = rand(4:10)
        lengths = (L, L)
        ΔL = L / N
        dt = 0.3 * rand()
        nsaves = rand(50:200)

        rs = range(; start=-L / 2, length=N, step=ΔL)
        u0 = (lg(rs, rs, l=rand(1:5)) + lg(rs, rs, l=rand(1:5)),)

        ts = 0:dt:nsaves*dt
        tspan = extrema(ts)

        sl_sol = free_propagation(u0[1], rs, rs, ts)

        dispersion1(ks, param) = sum(abs2, ks) / 2
        dispersion2(ks, param) = SVector(dispersion1(ks, param))
        dispersion3(ks, param) = SMatrix{1,1}(dispersion1(ks, param))

        alg = StrangSplitting()

        for dispersion in (dispersion1, dispersion2, dispersion3)
            prob = GrossPitaevskiiProblem(u0, lengths; dispersion)
            sol = solve(prob, alg, tspan; nsaves, dt, show_progress=false)[2]
            @test sol[1] ≈ sl_sol
        end
    end
end