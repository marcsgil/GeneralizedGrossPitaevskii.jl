module CUDAExt
using GeneralizedGrossPitaevskii, CUDA, CUDA.CUFFT, LinearAlgebra

"""
This extension is due to the fact that CUFFT is not well implemented in CUDA.jl.
See the issue https://github.com/JuliaGPU/CUDA.jl/issues/2641
"""

function GeneralizedGrossPitaevskii.get_fft_plans(u::CuArray{T1,N}, ru::CuArray{T2,N}, prob) where {T1,T2,N}
    ftdims = ntuple(identity, length(prob.lengths)) .+ (ndims(ru) - ndims(u))
    plan = plan_fft(ru, ftdims)
    iplan = inv(plan)
    plan, iplan
end

function GeneralizedGrossPitaevskii.get_fft_plans(u::CuArray{T1,M}, ru::CuArray{T2,N}, prob) where {T1,T2,M,N}
    ftdims = ntuple(identity, length(prob.lengths) + ndims(ru) - ndims(u)) 
    iftdims = ntuple(identity, ndims(ru) - ndims(u))
    plans = plan_fft(ru, ftdims), plan_ifft!(ru, iftdims)
    plans, inv.(plans)
end

function GeneralizedGrossPitaevskii.perform_ft!(dest::CuArray, plans::Tuple, src::CuArray)
    mul!(dest, plans[1], src)
    mul!(dest, plans[2], dest)
end

end