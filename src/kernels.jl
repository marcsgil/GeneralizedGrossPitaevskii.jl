@kernel function muladd_kernel!(dest::NTuple{N}, exp_δt, F_next, F_now, δt, noise_func, ξ, param) where {N}
    K = @index(Global, NTuple)

    fields = build_field_at(dest, K)

    exp_δt_val = _getindex(exp_δt, K)
    Fδt_next_val = _mul(δt / 2, _getindex(F_next, K))
    Fδt_now_val = _mul(δt / 2, _getindex(F_now, K))
    ξ_val = build_field_at(ξ, K)
    noise = _mul(-im * √δt, mul_noise(noise_func, ξ_val, fields, param))

    result = _add(_mul(exp_δt_val, _add(fields, Fδt_now_val)), Fδt_next_val)
    result = _add(result, noise)

    for (n, field) ∈ enumerate(dest)
        field[K...] = result[n]
    end
end

@kernel nonlinear_kernel!(::NTuple{N}, ::Nothing, param, δt) where {N} = nothing

@kernel function nonlinear_kernel!(dest::NTuple{N}, nonlinearity, param, δt) where {N}
    K = @index(Global)
    fields = SVector(getindex.(dest, K))
    result = _mul(_cis(-δt * nonlinearity(fields, param)), fields)
    for (n, field) ∈ enumerate(dest)
        field[K...] = result[n]
    end
end