abstract type AbstractAlgorithm end

abstract type FixedTimeSteppingAlgorithm <: AbstractAlgorithm end

struct StrangSplitting <: FixedTimeSteppingAlgorithm end

function diffusion_step!(iter)
    perform_ft!(iter.ft_buffer, iter.plan, iter.u)
    muladd_func!(iter.ft_buffer, iter.exp_Ddt, additiveIdentity, additiveIdentity, false, additiveIdentity,
        additiveIdentity, additiveIdentity, nothing; ndrange=size(first(fft_buffer)))
    perform_ft!(iter.u, iter.iplan, iter.fft_buffer)
end

function potential_pump_step!(t, dt, iter)
    prob = iter.prob
    sample_noise!(prob.noise_prototype, iter.rng)
    evaluate_pump!(prob, iter.pump_buffer_next, iter.pump_buffer_now, t)
    muladd_func!(iter.u, iter.exp_Vdt, iter.pump_buffer_next, iter.pump_buffer_now, dt,
        prob.nonlinearity, prob.noise_func, prob.noise_prototype, prob.param; ndrange=size(first(iter.u)))
end

function CommonSolve.step!(iter::GrossPitaevskiiIterator, t, dt)
    potential_pump_step!(t + dt / 2, dt / 2, iter)
    diffusion_step!(iter)
    potential_pump_step!(t + dt, dt / 2, iter)
end

function get_fft_plans(u, ::GrossPitaevskiiProblem{N,M}) where {N,M}
    ftdims = ntuple(identity, N)
    plan = plan_fft(first(u), ftdims)
    iplan = inv(plan)
    plan, iplan
end

struct GrossPitaevskiiIterator{PR,R,U,D,V,PB,PL,IPL,K,RNG}
    prob::PR
    result::R
    u::U
    ft_buffer::U
    exp_Ddt::D
    exp_Vdt::V
    pump_buffer_next::PB
    pump_buffer_now::PB
    plan::PL
    iplan::IPL
    kernel!::K
    rng::RGN
end

function CommonSolve.init(prob::GrossPitaevskiiProblem, ::StrangSplitting, tspan;
    dt, nsaves, save_start, workgroup_size, rng, kwargs...)
    result = map(prob.u0) do x
        stack(x for _ ∈ 1:nsaves+save_start)
    end

    u = copy.(prob.u0)
    ft_buffer = similar.(u)

    exp_Ddt = get_exponential(prob.dispersion, prob.u0, reciprocal_grid(prob), prob.param, dt)
    exp_Vdt = get_exponential(prob.potential, prob.u0, direct_grid(prob), prob.param, dt / 2)

    buffer_next = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, dt)
    buffer_now = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, dt)
    evaluate_pump!(prob, buffer_next, tspan[1])

    plan, iplan = get_fft_plans(u, prob)

    backend = get_backend(first(prob.u0))
    kernel! = muladd_kernel!(backend, workgroup_size...)

    GrossPitaevskiiIterator(prob, result, u, ft_buffer, exp_Ddt, exp_Vdt,
        buffer_next, buffer_now, plan, iplan, kernel!, rng)
end