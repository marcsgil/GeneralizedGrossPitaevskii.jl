@testset "get_exponential" begin
    f(x, param) = sum(abs2, x)
    f! = scalar2vector(f)
    xs = rand(ComplexF32, 128)
    u1 = similar(xs)
    exp1 = GeneralizedGrossPitaevskii.get_exponential(u1, f, (xs,), nothing, π)
    u2 = similar(xs, 1, length(xs))
    exp2 = GeneralizedGrossPitaevskii.get_exponential(u2, f!, (xs,), nothing, π)
    @test dropdims(exp2, dims=(1, 2)) ≈ exp1
end