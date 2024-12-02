using KernelAbstractions

function f(xs...)
    exp(-sum(abs2, xs))
end

@kernel function kernel!(dest, f, xs, ys)
    J = @index(Global, NTuple)
    dest[J...] = f(xs[J[1]], ys[J[2]])
end

function evaluate_on_grid!(dest, f, xs, ys)
    backend = get_backend(dest)
    func = kernel!(backend)
    func(dest, f, xs, ys; ndrange=size(dest))
end

@kernel function kernel_new!(dest::AbstractArray{T,N}, f, xs::Vararg{V,N}) where {T,V,N}
    J = @index(Global, NTuple)
    dest[J...] = f(ntuple(j -> xs[j][J[j]], N)...)
end

function evaluate_on_grid_new!(dest, f, xs...)
    backend = get_backend(dest)
    func = kernel_new!(backend)
    func(dest, f, xs...; ndrange=size(dest))
end
##
xs = LinRange(-3, 3, 512)
ys = copy(xs)

dest = Array{Float32}(undef, 512, 512)

evaluate_on_grid!(dest, f, xs, ys)
@benchmark evaluate_on_grid!($dest, $f, $xs, $ys)
##
dest_new = similar(dest)

evaluate_on_grid_new!(dest_new, f, xs, ys)

dest ≈ dest_new

@benchmark evaluate_on_grid_new!($dest_new, $f, $xs, $ys)