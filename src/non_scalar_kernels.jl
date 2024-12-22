# All nothing
@kernel muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{false}}, exp_type, dest, ::Nothing, ::Nothing, ::Nothing, δt) where {M,N,T} = nothing

# Just pump
@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{false}}, exp_type, dest, ::Nothing, F_next, F_now, δt) where {M,N,T}
    K = @index(Global, NTuple)
    fK = K[1:ndims(F_next)]
    dest[K...] += (F_next[fK...] + F_now[fK...]) * δt / 2
end

# Just dispersion/potential
@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{false}}, ::ScalarFunction, dest, exp_Dδt, ::Nothing, ::Nothing, δt) where {M,N,T}
    i, K... = @index(Global, NTuple)
    fK = K[1:ndims(exp_Dδt)]
    dest[i, K...] *= exp_Dδt[fK...]
end

@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{false}}, ::VectorFunction, dest, exp_Dδt, ::Nothing, ::Nothing, δt) where {M,N,T}
    i, K... = @index(Global, NTuple)
    fK = K[1:ndims(exp_Dδt)-1]
    dest[i, K...] *= exp_Dδt[i, fK...]
end

@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{false}}, ::MatrixFunction, dest, exp_Dδt, ::Nothing, ::Nothing, δt) where {M,N,T}
    i, K... = @index(Global, NTuple)
    fK = K[1::ndims(exp_Dδt)-2]

    tmp = zero(eltype(dest))
    for j ∈ axes(dest, 1)
        tmp += exp_Dδt[i, j, fK...] * dest[j, K...]
    end

    dest[i, K...] = tmp
end

# Dispersion/potential and pump
@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{false}}, ::ScalarFunction, dest, exp_Vδt, F_next, F_now, δt) where {M,N,T}
    i, K... = @index(Global, NTuple)
    fK = K[1:ndims(exp_Vδt)]
    dest[i, K...] = exp(exp_Vδt[fK...]) * (dest[i, K...] + F_now[i, fK...] * δt / 2) + F_next[i, fK...] * δt / 2
end

@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{false}}, ::VectorFunction, dest, exp_Vδt, F_next, F_now, δt) where {M,N,T}
    i, K... = @index(Global, NTuple)
    fK = K[1:ndims(exp_Vδt)-1]
    dest[i, K...] = exp(exp_Vδt[i, fK...]) * (dest[i, K...] + F_now[i, fK...] * δt / 2) + F_next[i, fK...] * δt / 2
end

@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{false}}, ::MatrixFunction, dest, exp_Vδt, F_next, F_now, δt) where {M,N,T}
    i, K... = @index(Global, NTuple)
    fK = K[1:ndims(exp_Vδt)-2]

    tmp = zero(eltype(dest))
    for j ∈ axes(dest, 1)
        tmp += exp_Vδt[i, j, fK...] * (dest[j, K...] + F_now[j, fK...] * δt / 2) + F_next[j, fK...] * δt / 2
    end

    dest[i, K...] = tmp
end