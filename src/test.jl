using AllocCheck

function f(xs...)
    exp(-sum(abs2, xs))
end

function evaluate_on_grid!(dest, f, xs, ys)
    for n ∈ eachindex(IndexCartesian(), dest)
        dest[n] = f(xs[n[1]], ys[n[2]])
    end
end

function evaluate_on_grid_new!(dest::AbstractArray{T,N}, f, xs...) where {T,N}
    for n ∈ eachindex(IndexCartesian(), dest)
        dest[n] = f(ntuple(j -> xs[j][n[j]], N)...)
    end
end
##
xs = LinRange(-3, 3, 128)
ys = copy(xs)

dest = Array{Float32}(undef, length(xs), length(ys))

evaluate_on_grid!(dest, f, xs, ys)

@benchmark evaluate_on_grid!($dest, $f, $xs, $ys)
##
dest_new = similar(dest)

evaluate_on_grid_new!(dest_new, f, xs, ys)

dest ≈ dest_new

@benchmark evaluate_on_grid_new!($dest_new, $f, $xs, $ys)