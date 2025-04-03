abstract type AbstractAlgorithm end

abstract type FixedTimeSteppingAlgorithm <: AbstractAlgorithm end

struct StrangSplitting <: FixedTimeSteppingAlgorithm end

function diffusion_step!(t, dt, u, prob, rng, fft_buffer, buffer_next, buffer_now, exp_Ddt, exp_Vdt,
    muladd_func!, plan, iplan)
    perform_ft!(fft_buffer, plan, u)
    muladd_func!(fft_buffer, exp_Ddt, AdditiveIdentity(), AdditiveIdentity(), false, AdditiveIdentity(),
        AdditiveIdentity(), AdditiveIdentity(), nothing; ndrange=size(first(fft_buffer)))
    perform_ft!(u, iplan, fft_buffer)
end

function potential_pump_step!(t, dt, u, prob, rng, fft_buffer, buffer_next, buffer_now, exp_Ddt, exp_Vdt,
    muladd_func!, plan, iplan)
    sample_noise!(prob.noise_prototype, rng)
    evaluate_pump!(prob, buffer_next, buffer_now, t)
    muladd_func!(u, exp_Vdt, buffer_next, buffer_now, dt, prob.nonlinearity, prob.noise_func, prob.noise_prototype, prob.param; ndrange=size(first(u)))
end

function step!(::StrangSplitting, t, dt, args...)
    potential_pump_step!(t + dt / 2, dt / 2, args...)
    diffusion_step!(nothing, nothing, args...)
    potential_pump_step!(t + dt, dt / 2, args...)
end

function get_fft_plans(u, ::GrossPitaevskiiProblem{N,M}) where {N,M}
    ftdims = ntuple(identity, N)
    plan = plan_fft(first(u), ftdims)
    iplan = inv(plan)
    plan, iplan
end

function get_precomputations(::StrangSplitting, prob, dt, tspan, nsaves, workgroup_size, save_start)
    result = map(prob.u0) do x
        stack(x for _ ∈ 1:nsaves+save_start)
    end

    u = fftshift.(prob.u0)
    fft_buffer = similar.(u)

    buffer_next = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, dt)
    buffer_now = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, dt)
    evaluate_pump!(prob, buffer_next, tspan[1])

    plan, iplan = get_fft_plans(u, prob)

    exp_Ddt = get_exponential(prob.dispersion, prob.u0, reciprocal_grid(prob), prob.param, dt)
    exp_Vdt = get_exponential(prob.potential, prob.u0, direct_grid(prob), prob.param, dt / 2)

    backend = get_backend(first(prob.u0))
    muladd_func! = muladd_kernel!(backend, workgroup_size...)

    result, u, fft_buffer, buffer_next, buffer_now, exp_Ddt, exp_Vdt,
    muladd_func!, plan, iplan
end

_copy!(::StrangSplitting, dest, src) = ifftshift!(dest, src)