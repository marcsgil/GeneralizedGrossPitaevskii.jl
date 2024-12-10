module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, ProgressMeter
using ExponentialUtilities, EllipsisNotation
using BenchmarkTools

include("misc.jl")
include("problem.jl")
export GrossPitaevskiiProblem, DiagonalNoise
include("strang_splitting_solver.jl")
export solve, StrangSplitting
include("templates.jl")
export free_propagation_template, kerr_propagation_template, lower_polariton_template

end
