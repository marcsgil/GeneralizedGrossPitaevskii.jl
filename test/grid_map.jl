@testset "grid_map" begin
    f(x, param) = sum(abs2, x)
    f! = scalar2vector(f)
    xs = rand(128)
    dest1 = similar(xs)
    GeneralizedGrossPitaevskii.grid_map!(dest1, f, (xs,), nothing)
    dest2 = similar(xs, 1, length(xs))
    GeneralizedGrossPitaevskii.grid_map!(dest2, f!, (xs,), nothing)
    @test dropdims(dest2, dims=1) == dest1
end