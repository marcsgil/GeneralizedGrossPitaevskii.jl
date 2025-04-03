struct AdditiveIdentity end
(::AdditiveIdentity)(args...) = AdditiveIdentity()

struct MultiplicativeIdentity end
(::MultiplicativeIdentity)(args...) = MultiplicativeIdentity()

_mul(x, y) = x * y
_mul(x::AbstractVector, y::AbstractVector) = x .* y
_mul(::MultiplicativeIdentity, y) = y
_mul(x, ::MultiplicativeIdentity) = x
_mul(::MultiplicativeIdentity, ::MultiplicativeIdentity) = MultiplicativeIdentity()
_mul(x, ::AdditiveIdentity) = AdditiveIdentity()
_mul(x, args...) = _mul(x, _mul(args...))

_add(x, y) = x .+ y
_add(::AdditiveIdentity, y) = y
_add(x, ::AdditiveIdentity) = x
_add(::AdditiveIdentity, ::AdditiveIdentity) = AdditiveIdentity()
_add(x, args...) = _add(x, _add(args...))

_cis(x) = cis(x)
_cis(x::AbstractVector) = cis.(x)
_cis(::AdditiveIdentity) = MultiplicativeIdentity()

_getindex(A, K) = A[K[1:ndims(A)]...]
_getindex(::T, K) where {T<:Union{AdditiveIdentity,MultiplicativeIdentity}} = T()

build_field_at(::T, K) where {T<:Union{AdditiveIdentity,MultiplicativeIdentity}} = T()
function build_field_at(ξ, K)
    SVector(map(ξ) do x
        _getindex(x, K)
    end)
end

@kernel function muladd_kernel!(dest::NTuple{N}, exp_δt, F_next, F_now, δt, nonlinearity, noise_func, ξ, param) where {N}
    K = @index(Global, NTuple)

    fields = build_field_at(dest, K)
    noise = _mul(-im * √δt, noise_func(fields, param), build_field_at(ξ, K))

    exp_δt_val = _mul(_cis(_mul(-δt, nonlinearity(fields, param))), _getindex(exp_δt, K))
    Fδt_next_val = _mul(δt / 2, _getindex(F_next, K))
    Fδt_now_val = _mul(δt / 2, _getindex(F_now, K))

    result = _add(_mul(exp_δt_val, _add(fields, Fδt_now_val)), Fδt_next_val)
    result = _add(result, noise)

    for (n, field) ∈ enumerate(dest)
        field[K...] = result[n]
    end
end