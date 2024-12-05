using Test, Random, StructuredLight, GeneralizedGrossPitaevskii

Random.seed!(1234)

scalar2vector(::Nothing) = nothing
scalar2vector(f) = (dest, args...) -> dest[1] = f(args...)
scalar2vector(g::Number) = reshape([g], 1, 1)

function scalar2vector(prob::GrossPitaevskiiProblem)
    new_u0 = reshape(prob.u0, 1, size(prob.u0)...)
    dispersion! = scalar2vector(prob.dispersion)
    potential! = scalar2vector(prob.potential)
    nonlinearity = scalar2vector(prob.nonlinearity)
    nonlinearity = GeneralizedGrossPitaevskii.to_device(prob.u0, nonlinearity)
    pump! = scalar2vector(prob.pump)
    GrossPitaevskiiProblem(dispersion!, potential!, nonlinearity, pump!, new_u0, prob.lengths, prob.param)
end

include("free_propagation.jl")
include("kerr_propagation.jl")
include("grid_map.jl")
include("get_exponential.jl")