using Test, Random, StructuredLight, GeneralizedGrossPitaevskii

Random.seed!(1234)

include("free_propagation.jl")
include("kerr_propagation.jl")
include("grid_map.jl")
include("get_exponential.jl")