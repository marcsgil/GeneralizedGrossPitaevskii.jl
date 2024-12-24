module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, ProgressMeter

using Reexport
@reexport using StaticArrays

include("problem.jl")
export GrossPitaevskiiProblem, AbstractGrossPitaevskiiProblem, EnsembleGrossPitaevskiiProblem

include("misc.jl")

include("potentials.jl")
export damping_potential

include("kernels.jl")

include("fixed_time_step_splitting_solvers.jl")
export solve, StrangSplittingA, StrangSplittingB, StrangSplittingC

end
