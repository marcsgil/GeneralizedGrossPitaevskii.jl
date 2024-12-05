function get_unionall(::T) where {T}
    getproperty(parentmodule(T), nameof(T))
end

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

function grid_map!(dest::AbstractArray{T,N}, f!, x, args...) where {T,N}
    grid_map!(dest, f!, (x,), args...)
end

grid_map!(dest, ::Nothing, rs, args...) = nothing

function damping_potential(x, xmin, xmax, width, peak)
    peak * (exp(-(x - xmin)^2 / width^2) + exp(-(x - xmax)^2 / width^2))
end