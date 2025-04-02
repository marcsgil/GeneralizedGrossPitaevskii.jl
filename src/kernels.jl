_cis(x) = cis(x)
_cis(x::AbstractVector) = cis.(x)
_cis(::AdditiveIdentity) = MultiplicativeIdentity()

_getindex(A, K) = A[K[1:ndims(A)]...]
_getindex(::Nothing, K) = nothing

_mul(x, y) = x * y
_mul(x::AbstractVector, y::AbstractVector) = x .* y
_mul(::MultiplicativeIdentity, y) = y
_mul(x, ::MultiplicativeIdentity) = x
#_mul(::Nothing, ::Nothing) = nothing
#_mul(x, ::Nothing) = nothing
#_mul(::Nothing, y) = y
#_mul(x, y, args...) = _mul(x, _mul(y, args...))

_add(x, y) = x .+ y
_add(x, ::Nothing) = x
_add(::Nothing, y) = nothing
_add(::Nothing, ::Nothing) = nothing
_add(x, y, args...) = _add(x, _add(y, args...))

other_mul(x, y) = x * y
other_mul(x::AbstractVector, y::AbstractVector) = x .* y
other_mul(::Nothing, ::Nothing) = nothing
other_mul(x, ::Nothing) = x
other_mul(::Nothing, y) = y

mul_noise(noise_func, ::Nothing, args...) = nothing
mul_noise(noise_func, ξ, args...) = _mul(noise_func(args...), ξ)

evaluate_or_nothing(::Nothing, args...) = nothing
evaluate_or_nothing(func, args...) = func(args...)

build_field_at(::Nothing, K) = nothing
function build_field_at(ξ, K)
    SVector(map(ξ) do x
        _getindex(x, K)
    end)
end

@kernel function muladd_kernel!(dest::NTuple{N}, exp_δt, F_next, F_now, δt, nonlinearity, noise_func, ξ, param) where {N}
    K = @index(Global, NTuple)

    fields = build_field_at(dest, K)

    exp_δt_val = other_mul(_cis(_mul(-δt, evaluate_or_nothing(nonlinearity, fields, param))), _getindex(exp_δt, K))
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