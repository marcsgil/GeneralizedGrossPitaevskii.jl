module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, ProgressMeter

using Reexport
@reexport using StaticArrays

include("function_types.jl")
export ScalarFunction, VectorFunction, MatrixFunction
include("problem.jl")
export GrossPitaevskiiProblem, AbstractGrossPitaevskiiProblem, EnsembleGrossPitaevskiiProblem

include("misc.jl")

include("potentials.jl")
export damping_potential

include("kernels.jl")
#include("scalar_kernels.jl")
#include("non_scalar_kernels.jl")
#include("non_linear_kernels.jl")

include("fixed_time_step_splitting_solvers.jl")
export solve, StrangSplittingA, StrangSplittingB, StrangSplittingC
include("templates.jl")
export free_propagation_template, kerr_propagation_template, lower_polariton_template

end
