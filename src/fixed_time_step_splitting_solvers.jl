abstract type GGPSolver end

abstract type StrangSplitting{T<:Real} <: GGPSolver end

struct StrangSplittingA{T<:Real} <: StrangSplitting{T}
    nsaves::Int
    δt::T
end

struct StrangSplittingB{T<:Real} <: StrangSplitting{T}
    nsaves::Int
    δt::T
end

struct StrangSplittingC{T<:Real} <: StrangSplitting{T}
    nsaves::Int
    δt::T
end

struct LieSplitting{T<:Real} <: GGPSolver
    nsaves::Int
    δt::T
end

get_exponential(prob, ::Nothing, grid, δt) = nothing

_exponential!(M::AbstractMatrix) = exponential!(M)
_exponential!(v::AbstractVector) = map!(exp, v, v)

function get_exponential(prob, f!::TensorFunction{T,N}, grid, δt) where {T,N}
    M = nsdims(prob)
    dest = Array{eltype(prob),M + N}(undef, ntuple(n -> size(prob.u0, 1), N)..., ssize(prob)...)

    function im_f!(dest, x, param)
        f!(dest, x, param)
        lmul!(-im * δt, dest)
    end

    grid_map!(dest, im_f!, grid, prob.param)
    for slice ∈ eachslice(dest, dims=ntuple(n -> n + N, M))
        _exponential!(slice)
    end

    to_device(prob.u0, dest)
end

function get_exponential(prob, f::ScalarFunction, grid, δt)
    dest = similar(prob.u0, ssize(prob)...)
    exp_im_f(x, param) = cis(-δt * f(x, param))
    grid_map!(dest, exp_im_f, grid, prob.param)
    dest
end

function diffusion_step!(prob, u, buffer, exp_Dδt, diffusion_func!, plan, iplan)
    mul!(buffer, plan, u)
    diffusion_func!(prob, prob.dispersion, buffer, exp_Dδt, nothing, nothing, nothing; ndrange=size(u))
    mul!(u, iplan, buffer)
end

function potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, prob, t, δt, muladd_func!)
    evaluate_pump!(prob, buffer_next, buffer_now, t)
    muladd_func!(prob, prob.potential, u, exp_Vδt, buffer_next, buffer_now, δt; ndrange=size(u))
end

function step!(u, buffer_next, buffer_now, fft_buffer, prob, ::StrangSplittingA, exp_Dδt, exp_Vδt, G_δt,
    muladd_func!, nonlinear_func!, plan, iplan, t, δt)

    diffusion_step!(prob, u, fft_buffer, exp_Dδt, muladd_func!, plan, iplan)
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, prob, t + δt / 2, δt / 2, muladd_func!)
    nonlinear_func!(u, G_δt; ndrange=ssize(prob))
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, prob, t + δt, δt / 2, muladd_func!)
    diffusion_step!(prob, u, fft_buffer, exp_Dδt, muladd_func!, plan, iplan)
end

function step!(u, buffer_next, buffer_now, fft_buffer, prob, ::StrangSplittingB, exp_Dδt, exp_Vδt, G_δt,
    muladd_func!, nonlinear_func!, plan, iplan, t, δt)

    diffusion_step!(prob, u, fft_buffer, exp_Dδt, muladd_func!, plan, iplan)
    nonlinear_func!(u, G_δt; ndrange=ssize(prob))
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, prob, t + δt, δt, muladd_func!)
    nonlinear_func!(u, G_δt; ndrange=ssize(prob))
    diffusion_step!(prob, u, fft_buffer, exp_Dδt, muladd_func!, plan, iplan)
end

function step!(u, buffer_next, buffer_now, fft_buffer, prob, ::StrangSplittingC, exp_Dδt, exp_Vδt, G_δt,
    muladd_func!, nonlinear_func!, plan, iplan, t, δt)

    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, prob, t + δt / 2, δt / 2, muladd_func!)
    nonlinear_func!(u, G_δt; ndrange=ssize(prob))
    diffusion_step!(prob, u, fft_buffer, exp_Dδt, muladd_func!, plan, iplan)
    nonlinear_func!(u, G_δt; ndrange=ssize(prob))
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, prob, t + δt, δt / 2, muladd_func!)
end

"""
    resolve_fixed_timestepping(solver::T, tspan) where {T<:GGPSolver}

This function resolves the fixed timestepping for the given solver and time span.
The problem is that the number of saves, time span and time step will, in general, not be compatible.
This functions resolves this by calculating a smaller time step `δt̅` such that everything fits.
It returns a Tuple containing a vector of time points `ts` (only the first entry is initialized `ts[1] = tspan[1]`), 
the number of steps per save `steps_per_save` and the new time step `δt̅`.
"""
function resolve_fixed_timestepping(solver::T, tspan) where {T<:GGPSolver}
    ts = Vector{typeof(solver.δt)}(undef, solver.nsaves + 1)
    ts[1] = tspan[1]

    ΔT = (tspan[2] - tspan[1]) / (solver.nsaves)
    steps_per_save = round(Int, ΔT / solver.δt, RoundUp)
    δt̅ = ΔT / steps_per_save

    ts, steps_per_save, δt̅
end

get_δt_combination(::StrangSplittingA, δt) = δt / 2, δt / 2, δt
get_δt_combination(::StrangSplittingB, δt) = δt / 2, δt, δt / 2
get_δt_combination(::StrangSplittingC, δt) = δt, δt / 2, δt / 2


function get_precomputations(prob, solver::StrangSplitting, tspan, δt)
    result = stack(prob.u0 for _ ∈ 1:solver.nsaves+1)

    u = ifftshift(prob.u0, sdims(prob))
    buffer_next = similar_or_nothing(u, prob.pump)
    evaluate_pump!(prob, buffer_next, tspan[1])
    buffer_now = similar_or_nothing(u, prob.pump)
    fft_buffer = similar(u)

    δts = get_δt_combination(solver, δt)
    exp_Dδt = get_exponential(prob, prob.dispersion, reciprocal_grid(prob), δts[1])
    exp_Vδt = get_exponential(prob, prob.potential, direct_grid(prob), δts[2])
    G_δt = mul_or_nothing(prob.nonlinearity, δts[3])

    plan = plan_fft(u, sdims(prob))
    iplan = inv(plan)

    backend = get_backend(prob.u0)
    muladd_func! = muladd_kernel!(backend)
    nonlinear_func! = nonlinear_kernel!(backend)

    result, u, buffer_next, buffer_now, fft_buffer, prob, solver, exp_Dδt, exp_Vδt, G_δt,
    muladd_func!, nonlinear_func!, plan, iplan
end

function solve(prob, solver::StrangSplitting, tspan; show_progress=true, fftw_num_threads=1)
    FFTW.set_num_threads(fftw_num_threads)

    ts, steps_per_save, δt̅ = resolve_fixed_timestepping(solver, tspan)

    result, u, args... = get_precomputations(prob, solver, tspan, δt̅)
    _result = @view result[ntuple(n -> :, ndims(prob.u0))..., begin+1:end]

    """return step!(u, args..., t, δt̅)"""

    """return @benchmark step!($u, $(args...), 0, 0)"""

    t = tspan[1]
    progress = Progress(steps_per_save * solver.nsaves)
    for (n, slice) ∈ enumerate(eachslice(_result, dims=ndims(_result)))
        for m ∈ 1:steps_per_save
            t += δt̅
            step!(u, args..., t, δt̅)
            _next!(progress, show_progress)
        end
        fftshift!(slice, u, sdims(prob))
        ts[n+1] = t
    end
    finish!(progress)

    ts, result
end