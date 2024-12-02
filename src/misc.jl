function get_unionall(::T) where {T}
    getproperty(parentmodule(T), nameof(T))
end

@kernel function grid_map_kernel!(dest::AbstractArray{T,N}, f!, x) where {T,N}
    J = ntuple(x -> axes(dest), N - 1)
    K = @index(Global, NTuple)
    slice = @view dest[J..., K...]
    f!(slice, x[J[1]])
end

@kernel function grid_map_kernel!(dest::AbstractArray{T,N}, f!, x, y) where {T,N}
    J = ntuple(x -> :, N - 2)
    K = @index(Global, NTuple)
    slice = @view dest[J..., K...]
    f!(slice, x[K[1]], y[K[2]])
end

function grid_map!(dest, f!, rs...)
    backend = get_backend(dest)
    func! = grid_map_kernel!(backend)
    func!(dest, f!, rs...; ndrange=size(dest)[end-length(rs)+1:end])
end