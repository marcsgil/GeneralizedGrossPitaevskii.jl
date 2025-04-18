using Preferences: set_preferences!
set_preferences!("GeneralizedGrossPitaevskii", "dispatch_doctor_mode" => "error")

using Test, Random, Logging, StructuredLight, GeneralizedGrossPitaevskii, FFTW, LinearAlgebra

Random.seed!(1234)

include("free_propagation.jl")
include("kerr_propagation.jl")
include("bistability_cycle.jl")
include("exciton_polariton_test.jl")
include("windowed_ft.jl")