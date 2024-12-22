using Test, Random, Logging, StructuredLight, GeneralizedGrossPitaevskii

Random.seed!(1234)

#include("test_problem.jl")
#include("free_propagation.jl")
#include("kerr_propagation.jl")
include("bistability_cycle.jl")