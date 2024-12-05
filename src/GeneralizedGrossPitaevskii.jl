module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, ProgressMeter
using ExponentialUtilities, EllipsisNotation
using BenchmarkTools

include("misc.jl")
include("problem.jl")
export GrossPitaevskiiProblem
include("strang_splitting_solver.jl")
export solve, StrangSplitting
include("stochastic_problem.jl")
export DiagonalNoise, StochasticGrossPitaevskiiProblem

end
