get_unionall(::Type{T}) where {T} = getproperty(parentmodule(T), nameof(T))
get_unionall(x) = get_unionall(typeof(x))
to_device(y, x) = get_unionall(y)(x)
to_device(y, ::Nothing) = nothing

@kernel function grid_map_kernel!(dest, f, grid, args...)
    K = @index(Global, NTuple)
    dest[K...] = f(ntuple(m -> grid[m][K[m]], length(grid)), args...)
end

function grid_map!(dest, f, grid, args...)
    backend = get_backend(dest)
    kernel! = grid_map_kernel!(backend)
    kernel!(dest, f, grid, args...; ndrange=size(dest))
end

grid_map!(dest, ::Nothing, grid, args...) = nothing

_exp(x) = exp(x)
_exp(x::AbstractVector) = exp.(x)
_cis(x) = cis(x)
_cis(x::AbstractVector) = cis.(x)

_getindex(A, K) = A[K[1:ndims(A)]...]
_getindex(::Nothing, K) = nothing

_mul(x, y) = x * y
_mul(x::AbstractVector, y::AbstractVector) = x .* y
_mul(::Nothing, ::Nothing) = nothing
_mul(x, ::Nothing) = nothing
_mul(::Nothing, y) = y
_mul(x, y, args...) = _mul(x, _mul(y, args...))

_add(x, y) = x .+ y
_add(x, ::Nothing) = x
_add(::Nothing, y) = nothing
_add(::Nothing, ::Nothing) = nothing
_add(x, y, args...) = _add(x, _add(y, args...))

mul_noise(noise_func, ::Nothing, args...) = nothing
mul_noise(noise_func, ξ, args...) = _mul(noise_func(args...), ξ)

function get_pump_buffer(pump, u, lengths, param, t)
    T = pump(lengths, param, t) |> typeof
    similar(u, T)
end

get_pump_buffer(::Nothing, args...) = nothing

function evaluate_pump!(prob::GrossPitaevskiiProblem{M,N,T,T1,T2,T3,T4,T5,Nothing},
    args...) where {M,N,T,T1,T2,T3,T4,T5}
    nothing
end

function evaluate_pump!(prob, dest, t)
    rs = direct_grid(prob)
    grid_map!(dest, prob.pump, rs, prob.param, t)
end

function evaluate_pump!(prob, dest_next, dest_now, t)
    copy!(dest_now, dest_next)
    evaluate_pump!(prob, dest_next, t)
end