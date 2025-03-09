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

get_exponential(::Nothing, u0, grid, param, δt) = nothing

function get_exponential(f, u0, grid, param, δt)
    cis_f(x, param) = _cis(-δt * f(x, param))
    T = cis_f(ntuple(m -> first(grid[m]), length(grid)), param) |> typeof
    dest = similar(first(u0), T, size(first(u0))[1:length(grid)])
    grid_map!(dest, cis_f, grid, param)
    dest
end

sample_noise!(::Nothing) = nothing
function sample_noise!(noise)
    for x ∈ noise
        randn!(x)
    end
end

function perform_ft!(dest, plan, src)
    for (_dest, _src) in zip(dest, src)
        mul!(_dest, plan, _src)
    end
end

function diffusion_step!(u, buffer, exp_Dδt, diffusion_func!, plan, iplan)
    perform_ft!(buffer, plan, u)
    diffusion_func!(buffer, exp_Dδt, nothing, nothing, zero(eltype(first(buffer))), nothing, nothing, nothing; ndrange=size(first(buffer)))
    perform_ft!(u, iplan, buffer)
end

function potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, prob, t, δt, muladd_func!)
    sample_noise!(ξ)
    evaluate_pump!(prob, buffer_next, buffer_now, t)
    muladd_func!(u, exp_Vδt, buffer_next, buffer_now, δt, prob.noise_func, ξ, prob.param; ndrange=size(first(u)))
end

function step!(u, fft_buffer, buffer_next, buffer_now, prob, ::StrangSplittingA, exp_Dδt, exp_Vδt, ξ,
    muladd_func!, nonlinear_func!, plan, iplan, t, δt)

    diffusion_step!(u, fft_buffer, exp_Dδt, muladd_func!, plan, iplan)
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, prob, t + δt / 2, δt / 2, muladd_func!)
    nonlinear_func!(u, prob.nonlinearity, prob.param, δt; ndrange=size(first(u)))
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, prob, t + δt, δt / 2, muladd_func!)
    diffusion_step!(u, fft_buffer, exp_Dδt, muladd_func!, plan, iplan)
end

function step!(u, fft_buffer, buffer_next, buffer_now, prob, ::StrangSplittingB, exp_Dδt, exp_Vδt, ξ,
    muladd_func!, nonlinear_func!, plan, iplan, t, δt)

    diffusion_step!(u, fft_buffer, exp_Dδt, muladd_func!, plan, iplan)
    nonlinear_func!(u, prob.nonlinearity, prob.param, δt / 2; ndrange=size(first(u)))
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, prob, t + δt, δt, muladd_func!)
    nonlinear_func!(u, prob.nonlinearity, prob.param, δt / 2; ndrange=size(first(u)))
    diffusion_step!(u, fft_buffer, exp_Dδt, muladd_func!, plan, iplan)
end

function step!(u, fft_buffer, buffer_next, buffer_now, prob, ::StrangSplittingC, exp_Dδt, exp_Vδt, ξ,
    muladd_func!, nonlinear_func!, plan, iplan, t, δt)

    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, prob, t + δt / 2, δt / 2, muladd_func!)
    nonlinear_func!(u, prob.nonlinearity, prob.param, δt / 2; ndrange=size(first(u)))
    diffusion_step!(u, fft_buffer, exp_Dδt, muladd_func!, plan, iplan)
    nonlinear_func!(u, prob.nonlinearity, prob.param, δt / 2; ndrange=size(first(u)))
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, prob, t + δt, δt / 2, muladd_func!)
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

reinterpret_or_nothing(::Nothing) = nothing
reinterpret_or_nothing(ξ) = reinterpret(reshape, eltype(eltype(ξ)), ξ)

function get_fft_plans(u, ::GrossPitaevskiiProblem{N,M}) where {N,M}
    ftdims = ntuple(identity, N)
    plan = plan_fft(first(u), ftdims)
    iplan = inv(plan)
    plan, iplan
end

function get_precomputations(prob, solver::StrangSplitting, tspan, δt, workgroup_size, save_start)
    result = map(prob.u0) do x
        stack(x for _ ∈ 1:solver.nsaves+save_start)
    end

    u = fftshift.(prob.u0)
    buffer_next = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, δt)
    buffer_now = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, δt)
    evaluate_pump!(prob, buffer_next, tspan[1])

    plan, iplan = get_fft_plans(u, prob)

    δts = get_δt_combination(solver, δt)
    exp_Dδt = get_exponential(prob.dispersion, prob.u0, reciprocal_grid(prob), prob.param, δts[1])
    exp_Vδt = get_exponential(prob.potential, prob.u0, direct_grid(prob), prob.param, δts[2])

    backend = get_backend(first(prob.u0))
    muladd_func! = muladd_kernel!(backend, workgroup_size...)
    nonlinear_func! = nonlinear_kernel!(backend, workgroup_size...)

    result, u, buffer_next, buffer_now, prob, solver, exp_Dδt, exp_Vδt, prob.noise_prototype,
    muladd_func!, nonlinear_func!, plan, iplan
end

function solve(prob::GrossPitaevskiiProblem{N,M}, solver::StrangSplitting, tspan;
    show_progress=true,
    save_start=true,
    fftw_num_threads=1,
    workgroup_size=(),
) where {N,M}
    FFTW.set_num_threads(fftw_num_threads)

    ts, steps_per_save, δt̅ = resolve_fixed_timestepping(solver, tspan)

    result, u, args... = get_precomputations(prob, solver, tspan, δt̅, workgroup_size, save_start)
    _result = map(result) do x
        @view x[ntuple(n -> :, ndims(first(result)) - 1)..., begin+save_start:end]
    end

    t = tspan[1]
    progress = Progress(steps_per_save * solver.nsaves)
    for n ∈ axes(first(_result), ndims(first(_result)))
        slice = map(_result) do x
            @view x[ntuple(n -> :, ndims(first(result)) - 1)..., n]
        end

        for _ ∈ 1:steps_per_save
            t += δt̅
            step!(u, slice, args..., t, δt̅)
            if show_progress
                next!(progress)
            end
        end
        for (dest, src) ∈ zip(slice, u)
            ifftshift!(dest, src)
        end
        ts[n+1] = t
    end
    finish!(progress)

    ts[begin+1-save_start:end], result
end