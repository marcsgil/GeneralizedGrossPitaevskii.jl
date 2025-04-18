module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, Random, ProgressMeter
import CommonSolve: solve, init, step!, solve!

using Reexport
@reexport using StaticArrays

include("problem.jl")
export GrossPitaevskiiProblem

include("kernels.jl")

include("misc.jl")

include("fixed_time_stepping.jl")
export solve

include("strang_splitting.jl")
export StrangSplitting

end
