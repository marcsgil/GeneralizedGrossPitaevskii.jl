module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, Random, ProgressMeter, CommonSolve

using Reexport
@reexport using StaticArrays

include("problem.jl")
export GrossPitaevskiiProblem

include("kernels.jl")

include("misc.jl")

include("strang_splitting.jl")
export StrangSplitting

include("fixed_time_stepping.jl")
export solve

end
