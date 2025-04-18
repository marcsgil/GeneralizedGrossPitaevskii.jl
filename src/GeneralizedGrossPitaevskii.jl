module GeneralizedGrossPitaevskii

using KernelAbstractions, FFTW, LinearAlgebra, Random, ProgressMeter
import CommonSolve: solve, init, step!, solve!

using Reexport
@reexport using StaticArrays

using DispatchDoctor: @stable
@stable default_mode = "disable" begin
    include("problem.jl")
    include("kernels.jl")
    include("misc.jl")
    include("fixed_time_stepping.jl")
    include("strang_splitting.jl")
end

export GrossPitaevskiiProblem, solve, StrangSplitting

end
