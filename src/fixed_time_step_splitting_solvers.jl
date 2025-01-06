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
    dest = similar(u0, T)
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

function step!(u, ru, buffer_next, buffer_now, fft_buffer, fft_rbuffer, prob, ::StrangSplittingA, exp_Dδt, exp_Vδt, G_δt, ξ, rξ,
    muladd_func!, nonlinear_func!, plan, iplan, t, δt)

    diffusion_step!(ru, fft_buffer, fft_rbuffer, exp_Dδt, muladd_func!, plan, iplan)
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, rξ, prob, t + δt / 2, δt / 2, muladd_func!)
    nonlinear_func!(u, G_δt; ndrange=size(u))
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, rξ, prob, t + δt, δt / 2, muladd_func!)
    diffusion_step!(ru, fft_buffer, fft_rbuffer, exp_Dδt, muladd_func!, plan, iplan)
end

function step!(u, ru, buffer_next, buffer_now, fft_buffer, fft_rbuffer, prob, ::StrangSplittingB, exp_Dδt, exp_Vδt, G_δt, ξ, rξ,
    muladd_func!, nonlinear_func!, plan, iplan, t, δt)

    diffusion_step!(ru, fft_buffer, fft_rbuffer, exp_Dδt, muladd_func!, plan, iplan)
    nonlinear_func!(u, G_δt; ndrange=size(u))
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, rξ, prob, t + δt, δt, muladd_func!)
    nonlinear_func!(u, G_δt; ndrange=size(u))
    diffusion_step!(ru, fft_buffer, fft_rbuffer, exp_Dδt, muladd_func!, plan, iplan)
end

function step!(u, ru, buffer_next, buffer_now, fft_buffer, fft_rbuffer, prob, ::StrangSplittingC, exp_Dδt, exp_Vδt, G_δt, ξ, rξ,
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

function get_precomputations(prob, solver::StrangSplitting, tspan, δt, reduction, workgroup_size, save_start)
    result = stack(reduction(prob.u0, prob.param) for _ ∈ 1:solver.nsaves+save_start)

    u = ifftshift(prob.u0)
    buffer_next = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, δt)
    buffer_now = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, δt)
    evaluate_pump!(prob, buffer_next, tspan[1])
    fft_buffer = similar(u)

    T = eltype(u)
    ru = reinterpret(reshape, eltype(T), u)
    fft_rbuffer = reinterpret(reshape, eltype(T), fft_buffer)

    ftdims = ntuple(identity, length(prob.lengths)) .+ (ndims(fft_rbuffer) - ndims(fft_buffer))
    plan = plan_fft(fft_rbuffer, ftdims)
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

    result, u, ru, buffer_next, buffer_now, fft_buffer, fft_rbuffer, prob, solver, exp_Dδt, exp_Vδt, G_δt, ξ, rξ,
    muladd_func!, nonlinear_func!, plan, iplan
end

function solve(prob, solver::StrangSplitting, tspan;
    show_progress=true,
    reduction=(sol, param) -> sol,
    save_start=true,
    fftw_num_threads=1,
    workgroup_size=())
    FFTW.set_num_threads(fftw_num_threads)

    ts, steps_per_save, δt̅ = resolve_fixed_timestepping(solver, tspan)

    result, u, args... = get_precomputations(prob, solver, tspan, δt̅, reduction, workgroup_size, save_start)
    _result = @view result[ntuple(n -> :, ndims(result) - 1)..., begin+save_start:end]
    @show size(_result)
    buffer = similar(prob.u0)

    t = tspan[1]
    progress = Progress(steps_per_save * solver.nsaves)
    for (n, slice) ∈ enumerate(eachslice(_result, dims=ndims(_result)))
        for m ∈ 1:steps_per_save
            t += δt̅
            step!(u, args..., t, δt̅)
            if show_progress
                next!(progress)
            end
        end
        fftshift!(buffer, u)
        slice .= reduction(buffer, prob.param)
        ts[n+1] = t
    end
    finish!(progress)

    ts[begin+1-save_start:end], result
end