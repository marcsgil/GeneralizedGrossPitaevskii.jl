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

get_exponential(::Nothing, u0, grid, param, δt) = nothing

function get_exponential(f, u0, grid, param, δt)
    cis_f(x, param) = _cis(-δt * f(x, param))
    T = cis_f(ntuple(m -> first(grid[m]), length(grid)), param) |> typeof
    dest = similar(first(u0), T, size(first(u0))[1:length(grid)])
    grid_map!(dest, cis_f, grid, param)
    dest
end

function get_pump_buffer(pump, u, lengths, param, t)
    T = pump(lengths, param, t) |> typeof
    similar(first(u), T, size(first(u))[1:length(lengths)])
end

get_pump_buffer(::Nothing, args...) = nothing

function evaluate_pump!(::GrossPitaevskiiProblem{N,M,T1,T2,T3,T4,T5,Nothing},
    args...) where {M,N,T1,T2,T3,T4,T5}
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

sample_noise!(::Nothing, rng) = nothing
function sample_noise!(noise, rng)
    for x ∈ noise
        randn!(rng, x)
    end
end

function perform_ft!(dest, plan, src)
    for (_dest, _src) in zip(dest, src)
        mul!(_dest, plan, _src)
    end
end
