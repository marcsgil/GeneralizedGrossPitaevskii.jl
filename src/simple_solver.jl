struct SimpleSolver{T} <: FixedTimeSteppingAlgorithm where {T}
    nsaves::Int
    δt::T
end

function get_precomputations(prob, solver::SimpleSolver, tspan, δt, workgroup_size, save_start)
    result = map(prob.u0) do x
        stack(x for _ ∈ 1:solver.nsaves+save_start)
    end

    u = fftshift.(prob.u0)
    buffer_next = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, δt)
    buffer_now = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, δt)
    evaluate_pump!(prob, buffer_next, tspan[1])

    plan, iplan = get_fft_plans(u, prob)
    exp_Dδt = get_exponential(prob.dispersion, prob.u0, reciprocal_grid(prob), prob.param, δt)
    exp_Vδt = get_exponential(prob.potential, prob.u0, direct_grid(prob), prob.param, δt)

    backend = get_backend(first(prob.u0))
    muladd_func! = muladd_kernel!(backend, workgroup_size...)
    nonlinear_func! = nonlinear_kernel!(backend, workgroup_size...)

    result, u, buffer_next, buffer_now, prob, solver, exp_Dδt, exp_Vδt, prob.noise_prototype,
    muladd_func!, nonlinear_func!, plan, iplan
end

function step!(u, fft_buffer, pump_buffer, pump_buffer2, prob, ::SimpleSolver, exp_Dδt, exp_Vδt, ξ,
    muladd_func!, nonlinear_func!, plan, iplan, t, δt, rng)

    mul!(fft_buffer[1], plan, u[1])
    fft_buffer[1] .*= exp_Dδt
    mul!(u[1], iplan, fft_buffer[1])

    u[1] .*= exp_Vδt

    nonlinear_func!(u, prob.nonlinearity, prob.param, δt; ndrange=size(first(u)))

    evaluate_pump!(prob, pump_buffer, t)
    @. u[1] += pump_buffer * δt

    if !isnothing(ξ)
        randn!(rng, ξ[1])
        grid_map!(pump_buffer, prob.noise_func, u, prob.param)
        @. u[1] += pump_buffer * ξ[1] * √δt
    end
end