function get_unionall(::T) where {T}
    getproperty(parentmodule(T), nameof(T))
end

@kernel function evaluate_on_grid_kernel!(dest, f!, params, rs...)
    M = ndims(dest)
    N = length(rs)
    J = ntuple(i -> axes(dest, i), M - N)
    K = @index(Global, NTuple)

    slice = @view dest[J..., K...]

    f!(slice, (r[k] for (k, r) in zip(K, rs))...; params)
end

@kernel function evaluate_on_grid_simple_kernel!(dest, f, params, xs)
    K = @index(Global, NTuple)
    dest[K...] = f(xs[K...]; params)
end

function evaluate_on_grid!(dest, f!, rs...; params=nothing)
    backend = get_backend(dest)
    func! = evaluate_on_grid_kernel!(backend)
    #T = get_unionall(dest)
    func!(dest, f!, params, rs...; ndrange=size(dest)[end-length(rs)+1:end])
end

function evaluate_on_grid_simple!(dest, f!, xs)
    backend = get_backend(dest)
    func! = evaluate_on_grid_simple_kernel!(backend)
    #T = get_unionall(dest)
    func!(dest, f!, params, xs; ndrange=size(dest))
end
##

function f!(dest, x; params=nothing)
    dest[1] = exp(-sum(abs2, x))
end

f_simple!(x; params=nothing) = exp(-abs2(x))
##
xs = LinRange(-3, 3, 512^2)
dest = Array{Float32}(undef, 1, 512^2)
dest_simple = Array{Float32}(undef, 512^2)

evaluate_on_grid!(dest, f!, xs)
evaluate_on_grid_simple!(dest_simple, f_simple!, xs)

dest[1, :] ≈ dest_simple
##

@code_warntype evaluate_on_grid!(dest, f!, xs)

@benchmark evaluate_on_grid!($dest, $f!, $xs)

@benchmark evaluate_on_grid_simple!($dest_simple, $f_simple!, $xs)