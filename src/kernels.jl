@kernel function muladd_kernel!(dest, exp_δt, F_next, F_now, δt, noise_func, ξ, param)
    K = @index(Global, NTuple)

    exp_δt_val = _getindex(exp_δt, K)
    Fδt_next_val = _mul(δt / 2, _getindex(F_next, K))
    Fδt_now_val = _mul(δt / 2, _getindex(F_now, K))
    ξ_val = _getindex(ξ, K)
    noise = _mul(-im * √δt, mul_noise(noise_func, ξ_val, dest[K...], param))

    dest[K...] = _add(_mul(exp_δt_val, _add(dest[K...], Fδt_now_val)), Fδt_next_val)
    dest[K...] = _add(dest[K...], noise)
end

@kernel nonlinear_kernel!(dest, ::Nothing, param, δt) = nothing

#= @kernel function nonlinear_kernel!(ψ, G_δt::AbstractVector)
    K = @index(Global)
    ψ[K] *= cis(-mapreduce((g, field) -> g * abs2(field), +, G_δt, ψ[K]))
end

@kernel function nonlinear_kernel!(ψ, G_δt)
    K = @index(Global)
    ψ[K] *= cis(-dot(ψ[K], G_δt, ψ[K]))
end =#

@kernel function nonlinear_kernel!(dest, nonlinearity, param, δt)
    K = @index(Global)
    dest[K] = _mul(_cis(-δt * nonlinearity(dest[K], param)), dest[K])
end