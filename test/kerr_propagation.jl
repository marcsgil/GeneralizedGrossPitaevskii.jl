@testset "Scalar Kerr Propagation" begin
    for n ∈ 1:5
        N = 64
        L = 10
        lengths = (L, L)
        ΔL = L / N
        dt = 0.002 * rand()
        nsaves = rand(50:200)
        g = randn()

        rs = range(; start=-L / 2, length=N, step=ΔL)
        u0 = (lg(rs, rs, l=rand(-5:5)) + lg(rs, rs, l=rand(-5:5)),)

        ts = 0:dt:nsaves*dt
        tspan = extrema(ts)

        sl_sol = kerr_propagation(u0[1], rs, rs, ts, nsaves; g=-g)

        dispersion1(ks, param) = sum(abs2, ks) / 2
        dispersion2(ks, param) = SVector(dispersion1(ks, param))
        dispersion3(ks, param) = SMatrix{1,1}(dispersion1(ks, param))

        nonlinearity1(ψ, param) = param.g * abs2.(ψ) / 2
        nonlinearity2(ψ, param) = SVector(nonlinearity1(ψ, param))
        nonlinearity3(ψ, param) = SMatrix{1,1}(nonlinearity1(ψ, param))

        param = (; g)
        alg = StrangSplitting()

        for dispersion in (dispersion1, dispersion2, dispersion3)
            for nonlinearity in (nonlinearity1, nonlinearity2, nonlinearity3)
                prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity, param)
                sol = solve(prob, alg, tspan; nsaves, dt, show_progress=false)[2]
                @test sol[1] ≈ sl_sol
            end
        end
    end
end