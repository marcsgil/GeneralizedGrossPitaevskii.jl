module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, ProgressMeter
using ExponentialUtilities, EllipsisNotation
using BenchmarkTools

include("misc.jl")
include("potentials.jl")
export damping_potential
include("problem.jl")
export GrossPitaevskiiProblem, AbstractGrossPitaevskiiProblem
include("kernels.jl")
include("fixed_time_step_splitting_solvers.jl")
export solve, StrangSplittingA, StrangSplittingB, StrangSplittingC
include("templates.jl")
export free_propagation_template, kerr_propagation_template, lower_polariton_template

end
