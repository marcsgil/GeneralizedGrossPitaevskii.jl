@testset "GrossPitaevskiiProblem Tests" begin
    # Define a test problem
    u0 = rand(ComplexF32, 4, 3, 2)
    lengths = (1, 2, 3)
    dispersion = 1.0
    potential = 1.0
    nonlinearity = 1.0
    pump = 1.0

    prob = GrossPitaevskiiProblem(u0, lengths, dispersion, potential, nonlinearity, pump)

    # Test size
    @test size(prob) == size(u0)
    @test size(prob, 1) == size(u0, 1)

    # Test ndims
    @test ndims(prob) == 3

    # Test eltype
    @test eltype(prob) == ComplexF32

    # Test nsdims
    @test GeneralizedGrossPitaevskii.nsdims(prob) == 3

    # Test sdims
    @test GeneralizedGrossPitaevskii.sdims(prob) == (1, 2, 3)

    # Test ssize
    @test GeneralizedGrossPitaevskii.ssize(prob) == (4, 3, 2)
    @test GeneralizedGrossPitaevskii.ssize(prob, 1) == 4

    # Test direct_grid
    direct_grid_result = GeneralizedGrossPitaevskii.direct_grid(prob)
    @test length(direct_grid_result) == 3
    @test length(direct_grid_result[1]) == 4
    @test eltype(direct_grid_result[1]) == Float64

    # Test reciprocal_grid
    reciprocal_grid_result = GeneralizedGrossPitaevskii.reciprocal_grid(prob)
    @test length(reciprocal_grid_result) == 3
    @test length(reciprocal_grid_result[1]) == 4
    @test eltype(reciprocal_grid_result[1]) == Float64
end