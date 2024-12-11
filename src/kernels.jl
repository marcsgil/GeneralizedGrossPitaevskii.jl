@kernel muladd_kernel!(dest, ::Nothing, ::Nothing) = nothing

@kernel function muladd_kernel!(dest, ::Nothing, drive)
    K = @index(Global, NTuple)
    dest[K..., ..] .+= drive[K...]
end

@kernel function muladd_kernel!(dest, A, ::Nothing)
    i, K... = @index(Global, NTuple)

    tmp = zero(eltype(dest))
    for j ∈ axes(dest, 1)
        tmp += A[i, j, K...] * dest[j, K...]
    end

    dest[i, K...] = tmp
end

@kernel function muladd_kernel!(dest::AbstractArray{T1,N}, A::AbstractArray{T2,N},
    ::Nothing) where {T1,T2,N}
    K = @index(Global, NTuple)
    dest[K..., ..] .*= A[K...]
end

@kernel function muladd_kernel!(dest, A, b)
    i, K... = @index(Global, NTuple)

    tmp = zero(eltype(dest))
    for j ∈ axes(dest, 1)
        tmp += A[i, j, K...] * (dest[j, K...] + b[j, K...])
    end

    dest[i, K...] = tmp
end

@kernel function muladd_kernel!(dest::AbstractArray{T1,N}, A::AbstractArray{T2,N},
    b) where {T1,T2,N}
    K = @index(Global, NTuple)
    dest[K..., ..] .= A[K...] * (dest[K..., ..] + b[K...])
end

@kernel nonlinear_kernel!(ψ, ::Nothing) = nothing

@kernel function nonlinear_kernel!(ψ, G_δt)
    K = @index(Global, NTuple)

    tmp = zero(eltype(ψ))
    for n ∈ axes(G_δt, 2), m ∈ axes(G_δt, 1)
        tmp -= G_δt[m, n] * conj(ψ[m, K...]) * ψ[n, K...]
    end

    for i ∈ axes(ψ, 1)
        ψ[i, K..., ..] *= cis(tmp)
    end
end

@kernel function nonlinear_kernel!(ψ, G_δt::Number)
    K = @index(Global, NTuple)
    ψ[K..., ..] *= cis(-G_δt * abs2(ψ[K...]))
end