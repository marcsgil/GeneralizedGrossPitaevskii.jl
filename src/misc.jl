function get_unionall(::T) where {T}
    getproperty(parentmodule(T), nameof(T))
end

@kernel function grid_map_kernel!(dest::AbstractArray{T,N}, f!, param, x) where {T,N}
    J = ntuple(x -> :, N - 1)
    K = @index(Global, NTuple)
    slice = @view dest[J..., K...]
    f!(slice, x[K[1]]; param)
end

@kernel function grid_map_kernel!(dest::AbstractArray{T,N}, f!, param, x, y) where {T,N}
    J = ntuple(x -> :, N - 2)
    K = @index(Global, NTuple)
    slice = @view dest[J..., K...]
    f!(slice, x[K[1]], y[K[2]]; param)
end

function grid_map!(dest, f!, rs...; param)
    backend = get_backend(dest)
    func! = grid_map_kernel!(backend)
    func!(dest, f!, param, rs...; ndrange=size(dest)[end-length(rs)+1:end])
end

grid_map!(dest, ::Nothing, rs...; param) = nothing