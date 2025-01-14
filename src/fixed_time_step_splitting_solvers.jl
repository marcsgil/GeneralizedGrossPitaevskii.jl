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

function perform_ft!(buffer1, buffer2, src, plan, perm, iperm)
    permutedims!(buffer1, src, perm)
    mul!(buffer2, plan, buffer1)
    permutedims!(src, buffer2, iperm)
    src, buffer1
end

function perform_ft!(buffer1, ::Nothing, src, plan, perm, iperm)
    mul!(buffer1, plan, src)
    buffer1, src
end

function dispersion_step!(u, ru, ft_buffer1, ft_buffer2, exp_Dδt, dispersion_func!, plan, iplan, perm, iperm)
    ft_result, new_buffer = perform_ft!(ft_buffer1, ft_buffer2, ru, plan, perm, iperm)
    ft_rresult = reinterpret(eltype(u), ft_result)
    dispersion_func!(ft_rresult, exp_Dδt, nothing, nothing, zero(eltype(ru)), nothing, nothing, nothing; ndrange=size(u))
    perform_ft!(new_buffer, ft_buffer2, ft_result, iplan, perm, iperm)
end

function potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, rξ, prob, t, δt, muladd_func!)
    sample_noise!(rξ)
    evaluate_pump!(prob, buffer_next, buffer_now, t)
    muladd_func!(u, exp_Vδt, buffer_next, buffer_now, δt, prob.noise_func, ξ, prob.param; ndrange=size(u))
end

function nonlinear_step!(u, prob, δt, nonlinear_func!)
    nonlinear_func!(u, prob.nonlinearity, prob.param, δt; ndrange=size(u))
end

function step!(u, ft_buffer1, ft_buffer2, plan, iplan, perm, iperm, ru, buffer_next, buffer_now, prob, ::StrangSplittingA, exp_Dδt, exp_Vδt, ξ, rξ,
    muladd_func!, nonlinear_func!, t, δt)

    dispersion_step!(u, ru, ft_buffer1, ft_buffer2, exp_Dδt, muladd_func!, plan, iplan, perm, iperm)
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, rξ, prob, t + δt / 2, δt / 2, muladd_func!)
    nonlinear_step!(u, prob, δt, nonlinear_func!)
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, rξ, prob, t + δt, δt / 2, muladd_func!)
    dispersion_step!(u, ru, ft_buffer1, ft_buffer2, exp_Dδt, muladd_func!, plan, iplan, perm, iperm)
end

function step!(u, ft_buffer1, ft_buffer2, plan, iplan, perm, iperm, ru, buffer_next, buffer_now, prob, ::StrangSplittingB, exp_Dδt, exp_Vδt, ξ, rξ,
    muladd_func!, nonlinear_func!, t, δt)

    dispersion_step!(u, ru, ft_buffer1, ft_buffer2, exp_Dδt, muladd_func!, plan, iplan, perm, iperm)
    nonlinear_step!(u, prob, δt / 2, nonlinear_func!)
    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, rξ, prob, t + δt, δt, muladd_func!)
    nonlinear_step!(u, prob, δt / 2, nonlinear_func!)
    dispersion_step!(u, ru, ft_buffer1, ft_buffer2, exp_Dδt, muladd_func!, plan, iplan, perm, iperm)
end

function step!(u, ft_buffer1, ft_buffer2, plan, iplan, perm, iperm, ru, buffer_next, buffer_now, prob, ::StrangSplittingC, exp_Dδt, exp_Vδt, ξ, rξ,
    muladd_func!, nonlinear_func!, t, δt)

    potential_pump_step!(u, buffer_next, buffer_now, exp_Vδt, ξ, rξ, prob, t + δt / 2, δt / 2, muladd_func!)
    nonlinear_step!(u, prob, δt / 2, nonlinear_func!)
    dispersion_step!(u, ru, ft_buffer1, ft_buffer2, exp_Dδt, muladd_func!, plan, iplan, perm, iperm)
    nonlinear_step!(u, prob, δt / 2, nonlinear_func!)
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

get_δt_combination(::StrangSplittingA, δt) = δt / 2, δt / 2
get_δt_combination(::StrangSplittingB, δt) = δt / 2, δt
get_δt_combination(::StrangSplittingC, δt) = δt, δt / 2

function get_exponentials(prob, solver::StrangSplitting, δt)
    δts = get_δt_combination(solver, δt)
    exp_Dδt = get_exponential(prob.dispersion, prob.u0, reciprocal_grid(prob), prob.param, δts[1])
    exp_Vδt = get_exponential(prob.potential, prob.u0, direct_grid(prob), prob.param, δts[2])
    exp_Dδt, exp_Vδt
end

function get_pump_buffers(prob, u, t₀, ::StrangSplitting)
    buffer_next = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, t₀)
    buffer_now = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, t₀)
    evaluate_pump!(prob, buffer_next, t₀)
    buffer_next, buffer_now
end

function get_fft_precomp(u, ru, prob)
    δ_ndims = ndims(ru) - ndims(u)
    first_dims = ntuple(identity, ndims(u)) .+ δ_ndims
    last_dims = ntuple(identity, δ_ndims)
    perm = first_dims..., last_dims...
    iperm = invperm(perm)

    ft_buffer1 = permutedims(ru, perm)
    ft_buffer2 = similar(ft_buffer1)

    ft_dims = ntuple(identity, length(prob.lengths))
    plan = plan_fft(ft_buffer1, ft_dims)
    iplan = inv(plan)
    ft_buffer1, ft_buffer2, plan, iplan, perm, iperm
end

function get_fft_precomp(::AbstractArray{T, N}, ru::AbstractArray{T, N}, prob) where {T, N}
    ft_buffer = similar(ru)
    ft_dims = ntuple(identity, length(prob.lengths))
    plan = plan_fft(ft_buffer, ft_dims)
    iplan = inv(plan)

    ft_buffer, nothing, plan, iplan, nothing, nothing
end

reinterpret_or_nothing(::Nothing) = nothing
reinterpret_or_nothing(ξ) = reinterpret(reshape, eltype(eltype(ξ)), ξ)

function get_precomputations(prob, solver, tspan, δt, workgroup_size, save_start)
    result = stack(prob.u0 for _ ∈ 1:solver.nsaves+save_start)

    u = ifftshift(prob.u0)
    ru = reinterpret(reshape, eltype(eltype(u)), u)

    exps = get_exponentials(prob, solver, δt)
    pump_buffers = get_pump_buffers(prob, u, tspan[1], solver)

    fft_precomp = get_fft_precomp(u, ru, prob)

    ξ = prob.noise_prototype
    rξ = reinterpret_or_nothing(ξ)

    backend = get_backend(prob.u0)
    muladd_func! = muladd_kernel!(backend, workgroup_size...)
    nonlinear_func! = nonlinear_kernel!(backend, workgroup_size...)

    result, u, fft_precomp..., ru, pump_buffers..., prob, solver, exps..., ξ, rξ,
    muladd_func!, nonlinear_func!
end

choose_ft_buffers(u, slice, ::Nothing, ::Nothing) = slice, reinterpret(reshape, eltype(eltype(u)), slice)
choose_ft_buffers(u, slice, buffer1, buffer2) = buffer1, buffer2

function solve(prob, solver::StrangSplitting, tspan;
    show_progress=true,
    save_start=true,
    fftw_num_threads=1,
    workgroup_size=(),
)
    FFTW.set_num_threads(fftw_num_threads)

    ts, steps_per_save, δt̅ = resolve_fixed_timestepping(solver, tspan)

    result, u, args... = get_precomputations(prob, solver, tspan, δt̅, workgroup_size, save_start)
    _result = @view result[ntuple(n -> :, ndims(result) - 1)..., begin+save_start:end]

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
        fftshift!(slice, u)
        ts[n+1] = t
    end
    finish!(progress)

    ts[begin+1-save_start:end], result
end