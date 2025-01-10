module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, Random, ProgressMeter, BenchmarkTools

using Reexport
@reexport using StaticArrays

include("problem.jl")
export GrossPitaevskiiProblem

include("misc.jl")

include("potentials.jl")
export damping_potential

include("kernels.jl")

include("fixed_time_step_splitting_solvers.jl")
export solve, StrangSplittingA, StrangSplittingB, StrangSplittingC

#TODO

# parallel parameters

end
