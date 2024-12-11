get_unionall(::Type{T}) where {T} = getproperty(parentmodule(T), nameof(T))
get_unionall(x) = get_unionall(typeof(x))
to_device(y, x) = get_unionall(y)(x)
to_device(y, ::Nothing) = nothing

@kernel function grid_map_kernel!(dest::AbstractArray{T,N}, f!, x::NTuple{M}, args...) where {T,N,M}
    J = ntuple(x -> :, N - M)
    K = @index(Global, NTuple)
    slice = @view dest[J..., K...]
    f!(slice, ntuple(m -> x[m][K[m]], M), args...)
end

@kernel function grid_map_kernel!(dest::AbstractArray{T,M}, f, x::NTuple{M}, args...) where {T,M}
    K = @index(Global, NTuple)
    dest[K...] = f(ntuple(m -> x[m][K[m]], M), args...)
end

function grid_map!(dest::AbstractArray{T,N}, f!, x::NTuple{M}, args...) where {T,N,M}
    backend = get_backend(dest)
    func! = grid_map_kernel!(backend)
    ndrange = size(dest)[1+N-M:N]
    func!(dest, f!, x, args...; ndrange)
end

grid_map!(dest, ::Nothing, x::NTuple{M}, args...) where {M} = nothing

mul_or_nothing(::Nothing, δt) = nothing
mul_or_nothing!(::Nothing, δt) = nothing
mul_or_nothing(x, δt) = x * δt
mul_or_nothing!(x, δt) = rmul!(x, δt)

similar_or_nothing(x, ::Nothing) = nothing
similar_or_nothing(x, _) = similar(x)

_next!(progress, show_progress) = show_progress ? next!(progress) : nothing