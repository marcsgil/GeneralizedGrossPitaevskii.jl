@kernel muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{true}}, exp_type ,dest, ::Nothing, ::Nothing, ::Nothing, δt) where {M,N,T} = nothing

@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{true}}, exp_type, dest, ::Nothing, F_next, F_now, δt) where {M,N,T}
    K = @index(Global, NTuple)
    fK = K[1:ndims(F_next)]
    dest[K...] += (F_next[fK...] + F_now[fK...]) * δt / 2
end

@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{true}}, exp_type, dest, exp_Dδt, ::Nothing, ::Nothing, δt) where {M,N,T}
    K = @index(Global, NTuple)
    fK = K[1:ndims(exp_Dδt)]
    dest[K...] *= exp_Dδt[fK...]
end

@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{true}}, exp_type, dest, exp_Vδt, F_next, F_now, δt) where {M,N,T}
    K = @index(Global, NTuple)
    fK = K[1:ndims(exp_Vδt)]
    dest[K...] = exp_Vδt[fK...] * (dest[K...] + F_now[fK...] * δt / 2) + F_next[fK...] * δt / 2
end