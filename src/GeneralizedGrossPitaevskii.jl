module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, ProgressMeter
using ExponentialUtilities, EllipsisNotation
using BenchmarkTools

include("misc.jl")
include("problem.jl")
include("strang_splitting_solver.jl")

export GrossPitaevskiiProblem, solve, StrangSplitting

end
