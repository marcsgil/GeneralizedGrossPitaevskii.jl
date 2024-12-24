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

mul_or_nothing(::Nothing, δt) = nothing
mul_or_nothing!(::Nothing, δt) = nothing
mul_or_nothing(x, δt) = x * δt
mul_or_nothing!(x, δt) = rmul!(x, δt)

similar_or_nothing(x, ::Nothing) = nothing
similar_or_nothing(x, _) = similar(x)

_next!(progress, show_progress) = show_progress ? next!(progress) : nothing

function evaluate_pump!(prob::GrossPitaevskiiProblem{N,T,T1,T2,T3,T4,T5,Nothing},
    args...) where {N,T,T1,T2,T3,T4,T5}
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