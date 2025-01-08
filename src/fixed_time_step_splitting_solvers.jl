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
    dest = similar(u0, T, size(u0)[1:length(grid)])
    grid_map!(dest, cis_f, grid, param)
    dest
end

sample_noise!(::Nothing) = nothing
sample_noise!(noise) = randn!(noise)

function diffusion_step!(ru, buffer, rbuffer, exp_Dδt, diffusion_func!, plan, iplan)
    mul!(rbuffer, plan, ru)
    diffusion_func!(buffer, exp_Dδt, nothing, nothing, zero(eltype(rbuffer)), nothing, nothing, nothing; ndrange=size(buffer))
    mul!(ru, iplan, rbuffer)
end

function potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, rξ, prob, t, δt, muladd_func!)
    sample_noise!(rξ)
    evaluate_pump!(prob, buffer_next, buffer_now, t)
    muladd_func!(u, exp_Vδt, buffer_next, buffer_now, δt, prob.noise_func, ξ, prob.param; ndrange=size(u))
end

function step!(u,  fft_buffer, fft_rbuffer, ru, buffer_next, buffer_now, prob, ::StrangSplittingA, exp_Dδt, exp_Vδt, G_δt, ξ, rξ,
    muladd_func!, nonlinear_func!, plan, iplan, t, δt)

    diffusion_step!(ru, fft_buffer, fft_rbuffer, exp_Dδt, muladd_func!, plan, iplan)
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, rξ, prob, t + δt / 2, δt / 2, muladd_func!)
    nonlinear_func!(u, G_δt; ndrange=size(u))
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, rξ, prob, t + δt, δt / 2, muladd_func!)
    diffusion_step!(ru, fft_buffer, fft_rbuffer, exp_Dδt, muladd_func!, plan, iplan)
end

function step!(u,  fft_buffer, fft_rbuffer, ru, buffer_next, buffer_now, prob, ::StrangSplittingB, exp_Dδt, exp_Vδt, G_δt, ξ, rξ,
    muladd_func!, nonlinear_func!, plan, iplan, t, δt)

    diffusion_step!(ru, fft_buffer, fft_rbuffer, exp_Dδt, muladd_func!, plan, iplan)
    nonlinear_func!(u, G_δt; ndrange=size(u))
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, rξ, prob, t + δt, δt, muladd_func!)
    nonlinear_func!(u, G_δt; ndrange=size(u))
    diffusion_step!(ru, fft_buffer, fft_rbuffer, exp_Dδt, muladd_func!, plan, iplan)
end

function step!(u,  fft_buffer, fft_rbuffer, ru, buffer_next, buffer_now, prob, ::StrangSplittingC, exp_Dδt, exp_Vδt, G_δt, ξ, rξ,
    muladd_func!, nonlinear_func!, plan, iplan, t, δt)

    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, rξ, prob, t + δt / 2, δt / 2, muladd_func!)
    nonlinear_func!(u, G_δt; ndrange=size(u))
    diffusion_step!(ru, fft_buffer, fft_rbuffer, exp_Dδt, muladd_func!, plan, iplan)
    nonlinear_func!(u, G_δt; ndrange=size(u))
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, rξ, prob, t + δt, δt / 2, muladd_func!)
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

function get_precomputations(prob, solver::StrangSplitting, tspan, δt, workgroup_size, save_start, reuse_u0)
    #= if reuse_u0
        @assert solver.nsaves == 1 && !save_start "In order to reuse `u0`, one must save only 1 point and set `save_start` to `false`."
        result = prob.u0
    else
        result = stack(prob.u0 for _ ∈ 1:solver.nsaves+save_start)
    end =#

    result = stack(prob.u0 for _ ∈ 1:solver.nsaves+save_start)

    u = ifftshift(prob.u0)
    buffer_next = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, δt)
    buffer_now = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, δt)
    evaluate_pump!(prob, buffer_next, tspan[1])

    ru = reinterpret(reshape, eltype(eltype(u)), u)

    ftdims = ntuple(identity, length(prob.lengths)) .+ (ndims(ru) - ndims(u))
    plan = plan_fft(ru, ftdims)
    iplan = inv(plan)

    δts = get_δt_combination(solver, δt)
    exp_Dδt = get_exponential(prob.dispersion, prob.u0, reciprocal_grid(prob), prob.param, δts[1])
    exp_Vδt = get_exponential(prob.potential, prob.u0, direct_grid(prob), prob.param, δts[2])
    G_δt = _mul(δts[3], prob.nonlinearity)

    ξ = prob.noise_prototype
    rξ = reinterpret_or_nothing(ξ)

    backend = get_backend(prob.u0)
    muladd_func! = muladd_kernel!(backend, workgroup_size...)
    nonlinear_func! = nonlinear_kernel!(backend, workgroup_size...)

    result, u, ru, buffer_next, buffer_now, prob, solver, exp_Dδt, exp_Vδt, G_δt, ξ, rξ,
    muladd_func!, nonlinear_func!, plan, iplan
end

function solve(prob, solver::StrangSplitting, tspan;
    show_progress=true,
    save_start=true,
    fftw_num_threads=1,
    workgroup_size=(),
    reuse_u0=false)
    FFTW.set_num_threads(fftw_num_threads)

    ts, steps_per_save, δt̅ = resolve_fixed_timestepping(solver, tspan)

    result, u, args... = get_precomputations(prob, solver, tspan, δt̅, workgroup_size, save_start, reuse_u0)
    _result = @view result[ntuple(n -> :, ndims(result) - 1)..., begin+save_start:end]

    t = tspan[1]
    progress = Progress(steps_per_save * solver.nsaves)
    for (n, slice) ∈ enumerate(eachslice(_result, dims=ndims(_result)))
        for m ∈ 1:steps_per_save
            t += δt̅
            rslice = reinterpret(reshape, eltype(eltype(u)), slice)
            step!(u, slice, rslice, args..., t, δt̅)
            if show_progress
                next!(progress)
            end
        end
        fftshift!(slice, u)
        ts[n+1] = t
    end
    finish!(progress)

    ts[begin+1-save_start:end], result
end