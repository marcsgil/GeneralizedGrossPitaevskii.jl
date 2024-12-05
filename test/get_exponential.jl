@testset "get_exponential" begin
    f(x...; param) = sum(abs2, x)
    f!(dest, x...; param) = dest[1] = f(x...; param)
    xs = rand(ComplexF32, 128)
    u1 = similar(xs)
    exp1 = GeneralizedGrossPitaevskii.get_exponential(f, u1, π, xs; param=nothing)
    u2 = similar(xs, 1, length(xs))
    exp2 = GeneralizedGrossPitaevskii.get_exponential(f!, u2, π, xs; param=nothing)
    @test dropdims(exp2, dims=(1, 2)) ≈ exp1
end