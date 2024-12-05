function get_unionall(::T) where {T}
    getproperty(parentmodule(T), nameof(T))
end

@kernel function grid_map_kernel!(dest::AbstractArray{T,N}, f!, param, x::Vararg{Any,M}) where {T,N,M}
    J = ntuple(x -> :, N - M)
    K = @index(Global, NTuple)
    slice = @view dest[J..., K...]
    f!(slice, ntuple(m -> x[m][K[m]], M)...; param)
end

@kernel function grid_map_kernel!(dest::AbstractArray{T,N}, param, x::Vararg{Any,N}) where {T,N}
    K = @index(Global, NTuple)
    dest[K...] = f(ntuple(m -> x[m][K[m]], M)...; param)
end

function grid_map!(dest::AbstractArray{T,N}, f!, x::Vararg{Any,M}; param) where {T,N,M}
    backend = get_backend(dest)
    func! = grid_map_kernel!(backend)
    ndrange = size(dest)[1+N-M:N]
    func!(dest, f!, param, x...; ndrange)
end

grid_map!(dest, ::Nothing, rs...; param) = nothing