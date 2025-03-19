struct SimpleAlg <: FixedTimeSteppingAlgorithm end

function get_precomputations(::SimpleAlg, prob, dt, tspan, nsaves, workgroup_size, save_start)
    result = map(prob.u0) do x
        stack(x for _ ∈ 1:nsaves+save_start)
    end

    u = fftshift.(prob.u0)
    fft_buffer = similar.(u)

    pump_buffer = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, dt)
    evaluate_pump!(prob, pump_buffer, tspan[1])

    plan, iplan = get_fft_plans(u, prob)
    exp_Ddt = get_exponential(prob.dispersion, prob.u0, reciprocal_grid(prob), prob.param, dt)
    exp_Vdt = get_exponential(prob.potential, prob.u0, direct_grid(prob), prob.param, dt)

    backend = get_backend(first(prob.u0))
    nonlinear_func! = nonlinear_kernel!(backend, workgroup_size...)

    result, u, fft_buffer, pump_buffer, exp_Ddt, exp_Vdt, nonlinear_func!, plan, iplan
end

function step!(::SimpleAlg, t, dt, u, prob, rng, fft_buffer, pump_buffer, exp_Ddt, exp_Vdt, nonlinear_func!, plan, iplan)
    mul!(fft_buffer[1], plan, u[1])
    fft_buffer[1] .*= exp_Ddt
    mul!(u[1], iplan, fft_buffer[1])

    if !isnothing(exp_Vdt)
        u[1] .*= exp_Vdt
    end

    nonlinear_func!(u, prob.nonlinearity, prob.param, dt; ndrange=size(first(u)))

    evaluate_pump!(prob, pump_buffer, t)

    if !isnothing(prob.pump)
        @. u[1] += pump_buffer * dt
    end

    if !isnothing(prob.noise_prototype)
        randn!(rng, prob.noise_prototype[1])
        grid_map!(pump_buffer, prob.noise_func, u, prob.param)
        @. u[1] += pump_buffer * prob.noise_prototype[1] * √dt
    end
end