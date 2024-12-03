module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, ProgressMeter
using ExponentialUtilities, EllipsisNotation

include("misc.jl")
include("problem.jl")
include("solver.jl")

export GrossPitaevskiiProblem, solve

end
