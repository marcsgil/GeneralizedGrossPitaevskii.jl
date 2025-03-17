module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, Random, ProgressMeter

using Reexport
@reexport using StaticArrays

include("problem.jl")
export GrossPitaevskiiProblem

include("misc.jl")

include("kernels.jl")

include("fixed_time_step_splitting_solvers.jl")
export solve, StrangSplittingA, StrangSplittingB, StrangSplittingC

end
