@kernel muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{false}}, dest, ::Nothing, ::Nothing, ::Nothing, δt) where {M,N,T} = nothing
@kernel muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{true}}, dest, ::Nothing, ::Nothing, ::Nothing, δt) where {M,N,T} = nothing

@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{false}}, dest, ::Nothing, F_next, F_now, δt) where {M,N,T}
    K = @index(Global, NTuple)
    fK = K[1:M]
    dest[K...] += (F_next[fK...] + F_now[fK...]) * δt / 2
end

@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{true}}, dest, ::Nothing, F_next, F_now, δt) where {M,N,T}
    K = @index(Global, NTuple)
    fK = K[1:M]
    dest[K...] += (F_next[fK...] + F_now[fK...]) * δt / 2
end

@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{false}}, dest, exp_Dδt, ::Nothing, ::Nothing, δt) where {M,N,T}
    i, K... = @index(Global, NTuple)
    fK = K[1:M]

    tmp = zero(eltype(dest))
    for j ∈ axes(dest, 1)
        tmp += exp_Dδt[i, j, fK...] * dest[j, K...]
    end

    dest[i, K...] = tmp
end

@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{false}}, dest, exp_Vδt, F_next, F_now, δt) where {M,N,T}
    i, K... = @index(Global, NTuple)
    fK = K[1:M]

    tmp = zero(eltype(dest))
    for j ∈ axes(dest, 1)
        tmp += exp_Vδt[i, j, fK...] * (dest[j, K...] + F_now[j, fK...] * δt / 2) + F_next[j, fK...] * δt / 2
    end

    dest[i, K...] = tmp
end

@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{true}}, dest, A, ::Nothing, ::Nothing, δt) where {M,N,T}
    K = @index(Global, NTuple)
    fK = K[1:M]
    dest[K...] *= A[fK...]
end

@kernel function muladd_kernel!(::AbstractGrossPitaevskiiProblem{M,N,T,Val{true}}, dest, exp_Vδt, F_next, F_now, δt) where {M,N,T}
    K = @index(Global, NTuple)
    fK = K[1:M]
    dest[K...] = exp_Vδt[fK...] * (dest[K...] + F_now[fK...] * δt / 2) + F_next[fK...] * δt / 2
end

@kernel nonlinear_kernel!(ψ, ::Nothing) = nothing

@kernel function nonlinear_kernel!(ψ, G_δt)
    K = @index(Global, NTuple)

    tmp = zero(eltype(ψ))
    for n ∈ axes(G_δt, 2), m ∈ axes(G_δt, 1)
        tmp -= G_δt[m, n] * conj(ψ[m, K...]) * ψ[n, K...]
    end

    for i ∈ axes(ψ, 1)
        ψ[i, K...] *= cis(tmp)
    end
end

@kernel function nonlinear_kernel!(ψ, G_δt::Number)
    K = @index(Global)
    ψ[K] *= cis(-G_δt * abs2(ψ[K]))
end