_mul(x, y) = x * y
_mul(x::AbstractVector, y) = x .* y

@kernel muladd_kernel!(dest, ::Nothing, ::Nothing, ::Nothing, δt) = nothing

@kernel function muladd_kernel!(dest, ::Nothing, F_next, F_now, δt)
    K = @index(Global, NTuple)
    K_f = K[1:ndims(F_next)]
    dest[K...] = dest[K...] .+ (F_next[K_f...] + F_now[K_f...]) * δt / 2
end

@kernel function muladd_kernel!(dest, exp_Dδt, ::Nothing, ::Nothing, δt)
    K = @index(Global, NTuple)
    K_D = K[1:ndims(exp_Dδt)]
    dest[K...] = _mul(exp_Dδt[K_D...], dest[K...])
end

@kernel function muladd_kernel!(dest, exp_Vδt, F_next, F_now, δt)
    K = @index(Global, NTuple)
    K_V = K[1:ndims(exp_Vδt)]
    K_f = K[1:ndims(F_next)]
    dest[K...] = _mul(exp_Vδt[K_V...], (dest[K...] .+ F_now[K_f...] * δt / 2)) .+ F_next[K_f...] * δt / 2
end

@kernel nonlinear_kernel!(ψ, ::Nothing) = nothing

@kernel function nonlinear_kernel!(ψ, G_δt::AbstractVector)
    K = @index(Global)
    ψ[K] *= cis(-mapreduce((g, field) -> g * abs2(field), +, G_δt, ψ[K]))
end

@kernel function nonlinear_kernel!(ψ, G_δt)
    K = @index(Global)
    ψ[K] *= cis(-dot(ψ[K], G_δt, ψ[K]))
end