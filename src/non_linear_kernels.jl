@kernel nonlinear_kernel!(ψ, ::Nothing) = nothing

@kernel function nonlinear_kernel!(ψ, G_δt::Number)
    K = @index(Global)
    ψ[K] *= cis(-G_δt * abs2(ψ[K]))
end

@kernel function nonlinear_kernel!(ψ, G_δt::AbstractVector)
    K = @index(Global, NTuple)

    for i ∈ axes(ψ, 1)
        ψ[i, K...] *= cis(-G_δt[i] * abs2(ψ[i, K...]))
    end
end

@kernel function nonlinear_kernel!(ψ, G_δt::AbstractMatrix)
    K = @index(Global, NTuple)

    tmp = zero(eltype(ψ))
    for n ∈ axes(G_δt, 2), m ∈ axes(G_δt, 1)
        tmp -= G_δt[m, n] * conj(ψ[m, K...]) * ψ[n, K...]
    end

    for i ∈ axes(ψ, 1)
        ψ[i, K...] *= cis(tmp)
    end
end