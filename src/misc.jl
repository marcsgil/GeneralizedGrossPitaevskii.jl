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

function damping_potential_base(x, xmin, xmax, width)
    exp(-(x - xmin)^2 / width^2) + exp(-(x - xmax)^2 / width^2)
end

function damping_potential(x::NTuple{1}, xmin, xmax, widht)
    -im * damping_potential_base(x[1], xmin, xmax, widht)
end

"""pot_reduction(Vs) = 2 * sum(Vs) / (1 + prod(Vs)) / length(Vs)

first_or_getindex(x::Number, inds...) = x
first_or_getindex(x, inds...) = x[inds...]"""

"""
    damping_potential(x, xmin, xmax, width, Vmax=1)

Calculate a damping potential for a given position `x` and parameters `xmin`, `xmax`, `width` and `Vmax`.
The potential is approximatelly zero at the center of the grid and grows towards the boundaries.
"""

"""function damping_potential(x::NTuple{N}, xmin, xmax, width, Vmax=1) where {N}
    Vs = ntuple(i -> damping_potential_base(x[i],
            first_or_getindex(xmin, i),
            first_or_getindex(xmax, i),
            first_or_getindex(width, i)), N)
    Vmax * pot_reduction(Vs)
end

function damping_potential(x::Number, xmin, xmax, width, Vmax=1)
    damping_potential((x,), xmin, xmax, width, Vmax)
end"""

smooth_min(x, y; β=5) = ((x + y) - √((x - y)^2 + 1 / β)) / 2
smooth_min(x, y, args...; β=5) = smooth_min(smooth_min(x, y; β), args...; β)

"""function damping_potential(x, xmin, xmax, width)
    dist_min = smooth_min(abs.(x .- xmin)..., abs.(x .- xmax)...)
    -im * exp(-dist_min^2 / width^2)
end"""