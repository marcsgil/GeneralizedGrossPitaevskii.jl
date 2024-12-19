@kernel muladd_kernel!(dest, ::Nothing, ::Nothing, ::Nothing, δt) = nothing

@kernel function muladd_kernel!(dest, ::Nothing, F_next, F_now, δt)
    K = @index(Global, NTuple)
    dest[K...] .+= (F_next[K...] + F_next[K...]) * δt / 2
end

@kernel function muladd_kernel!(dest, exp_Dδt, ::Nothing, ::Nothing, δt)
    i, K... = @index(Global, NTuple)

    tmp = zero(eltype(dest))
    for j ∈ axes(dest, 1)
        tmp += exp_Dδt[i, j, K...] * dest[j, K...]
    end

    dest[i, K...] = tmp
end

@kernel function muladd_kernel!(dest::AbstractArray{T1,N}, A::AbstractArray{T2,N},
    ::Nothing, ::Nothing, δt) where {T1,T2,N}
    K = @index(Global, NTuple)
    dest[K...] .*= A[K...]
end

@kernel function muladd_kernel!(dest, exp_Vδt, F_next, F_now, δt)
    i, K... = @index(Global, NTuple)

    tmp = zero(eltype(dest))
    for j ∈ axes(dest, 1)
        tmp += exp_Vδt[i, j, K...] * (dest[j, K...] + F_now[j, K...] * δt / 2) + F_next[j, K...] * δt / 2
    end

    dest[i, K...] = tmp
end

@kernel function muladd_kernel!(dest::AbstractArray{T1,N}, exp_Vδt::AbstractArray{T2,N},
    F_next, F_now, δt) where {T1,T2,N}
    K = @index(Global, NTuple)
    dest[K...] .= exp_Vδt[K...] * (dest[K..., ..] + F_now[K...] * δt / 2) + F_next[K...] * δt / 2
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