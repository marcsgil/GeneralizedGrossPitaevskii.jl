module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, Random, ProgressMeter, CommonSolve
import CommonSolve: solve, step!
export solve

using Reexport
@reexport using StaticArrays

include("problem.jl")
export GrossPitaevskiiProblem

include("kernels.jl")

include("misc.jl")

include("fixed_time_stepping.jl")

include("strang_splitting.jl")
export StrangSplitting

end
