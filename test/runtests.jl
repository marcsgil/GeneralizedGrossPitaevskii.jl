using Test, Random, Logging, StructuredLight, GeneralizedGrossPitaevskii

Random.seed!(1234)

include("free_propagation.jl")
include("kerr_propagation.jl")
include("bistability_cycle.jl")
include("exciton_polariton_test.jl")