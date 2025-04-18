struct StrangSplitting <: FixedTimeSteppingAlgorithm end

struct StrangSplittingIterator{PROB,T,PROG1,PROG2,R,U,D,V,PB,PL,IPL,K,RNG} <:FixedTimeSteppingIterator
    prob::PROB
    dt::T
    ts::Vector{T}
    steps_per_save::Int
    save_start::Bool
    progress::PROG1
    given_progress::PROG2
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
    rng::RNG
end

function init(prob::GrossPitaevskiiProblem, ::StrangSplitting, tspan;
    dt,
    nsaves,
    show_progress=true,
    progress=nothing,
    save_start=true,
    workgroup_size=(),
    rng=nothing)

    result = map(prob.u0) do x
        stack(x for _ ∈ 1:nsaves+save_start)
    end

    dt, ts, steps_per_save = resolve_fixed_timestepping(dt, tspan, nsaves)
    _progress = _Progress(progress, steps_per_save * nsaves; enabled=show_progress)

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

    StrangSplittingIterator(prob, dt, ts, steps_per_save, save_start, _progress, progress, result, u, ft_buffer, exp_Ddt, exp_Vdt,
        buffer_next, buffer_now, plan, iplan, kernel!, rng)
end

function diffusion_step!(iter)
    perform_ft!(iter.ft_buffer, iter.plan, iter.u)
    iter.kernel!(iter.ft_buffer, iter.exp_Ddt, additiveIdentity, additiveIdentity, false, additiveIdentity,
        additiveIdentity, additiveIdentity, nothing; ndrange=size(first(iter.ft_buffer)))
    perform_ft!(iter.u, iter.iplan, iter.ft_buffer)
end

function potential_pump_step!(t, dt, iter)
    prob = iter.prob
    sample_noise!(prob.noise_prototype, iter.rng)
    evaluate_pump!(prob, iter.pump_buffer_next, iter.pump_buffer_now, t)
    iter.kernel!(iter.u, iter.exp_Vdt, iter.pump_buffer_next, iter.pump_buffer_now, dt,
        prob.nonlinearity, prob.noise_func, prob.noise_prototype, prob.param; ndrange=size(first(iter.u)))
end

function step!(iter::StrangSplittingIterator, t, dt)
    potential_pump_step!(t + dt / 2, dt / 2, iter)
    diffusion_step!(iter)
    potential_pump_step!(t + dt, dt / 2, iter)
end