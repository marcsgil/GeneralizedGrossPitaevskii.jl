module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, ProgressMeter
using ExponentialUtilities, EllipsisNotation

include("misc.jl")
include("problem.jl")
include("strang_splitting_solver.jl")

export GrossPitaevskiiProblem, solve, StrangSplitting

end
