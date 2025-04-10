module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, Random, ProgressMeter

using Reexport
@reexport using StaticArrays

include("problem.jl")
export GrossPitaevskiiProblem

include("kernels.jl")

include("misc.jl")

include("strang_splitting.jl")
export StrangSplitting

include("simple_alg.jl")

include("exp_finite_dif.jl")

include("fixed_time_stepping.jl")
export solve

end
