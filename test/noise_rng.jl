# A dispatch probe verifies backend selection without requiring GPU hardware.
struct BackendNoiseProbe
    calls::Base.RefValue{Int}
end

function Random.randn!(x::BackendNoiseProbe)
    x.calls[] += 1
    x
end

@testset "Noise RNG selection" begin
    GGP = GeneralizedGrossPitaevskii
    noise_func(u, r, p) = 1.0
    prob = GrossPitaevskiiProblem((zeros(ComplexF64, 4),), (1.0,);
        position_noise_func=noise_func, noise_prototype=(zeros(ComplexF64, 4),))
    iter = GGP.init(prob, StrangSplitting(), (0.0, 0.2);
        dt=0.1, nsaves=1, show_progress=false)
    @test iter.rng === nothing

    probe = BackendNoiseProbe(Ref(0))
    GGP.sample_noise!(noise_func, (probe,), iter.rng)
    @test probe.calls[] == 1

    rng = MersenneTwister(42)
    reference_rng = copy(rng)
    actual = zeros(ComplexF64, 4)
    expected = similar(actual)
    GGP.sample_noise!(noise_func, (actual,), rng)
    randn!(reference_rng, expected)
    @test actual == expected

    _, first_sol = solve(prob, StrangSplitting(), (0.0, 0.2);
        dt=0.1, nsaves=1, show_progress=false, rng=MersenneTwister(42))
    _, second_sol = solve(prob, StrangSplitting(), (0.0, 0.2);
        dt=0.1, nsaves=1, show_progress=false, rng=MersenneTwister(42))
    @test first_sol == second_sol
end
