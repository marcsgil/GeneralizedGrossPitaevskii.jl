module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, Random, ProgressMeter

using Reexport
@reexport using StaticArrays

include("problem.jl")
export GrossPitaevskiiProblem

include("misc.jl")

include("kernels.jl")

include("strang_splitting.jl")
export StrangSplittingA, StrangSplittingB, StrangSplittingC

include("simple_alg.jl")
export SimpleAlg

include("exp_finite_dif.jl")
export ExpFiniteDiff

include("fixed_time_stepping.jl")
export solve

end
