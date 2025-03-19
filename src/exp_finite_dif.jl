struct ExpFiniteDiff <: FixedTimeSteppingAlgorithm end

get_unionall(::Type{T}) where {T} = getproperty(parentmodule(T), nameof(T))
get_unionall(x) = get_unionall(typeof(x))
to_device(y, x) = get_unionall(y)(x)
to_device(y, ::Nothing) = nothing

function get_precomputations(::ExpFiniteDiff, prob, dt, tspan, nsaves, workgroup_size, save_start)
    result = map(prob.u0) do x
        stack(x for _ ∈ 1:nsaves+save_start)
    end

    u = copy.(prob.u0)
    finite_diff_buffer = similar.(u)

    pump_buffer1 = get_pump_buffer(prob.pump, u, prob.lengths, prob.param, dt)
    pump_buffer2 = similar(pump_buffer1)

    exp_Ddt = to_device(first(prob.u0), cis(-prob.dispersion(prob) * dt))

    #rs = map(xs -> to_device(first(prob.u0), xs), direct_grid(prob))
    #@show typeof(rs[1])
    exp_Vdt = get_exponential(prob.potential, prob.u0, direct_grid(prob), prob.param, dt)

    backend = get_backend(first(prob.u0))
    nonlinear_func! = nonlinear_kernel!(backend, workgroup_size...)

    result, u, finite_diff_buffer, pump_buffer1, pump_buffer2, exp_Ddt, exp_Vdt, nonlinear_func!
end

function step!(::ExpFiniteDiff, t, dt, u, prob, rng, finite_diff_buffer, pump_buffer1, pump_buffer2, exp_Ddt, exp_Vdt, nonlinear_func!)
    mul!(finite_diff_buffer[1], exp_Ddt, u[1])

    if !isnothing(exp_Vdt)
        u[1] .= finite_diff_buffer[1] .* exp_Vdt
    end

    nonlinear_func!(u, prob.nonlinearity, prob.param, dt; ndrange=size(first(u)))

    grid_map!(pump_buffer1, prob.pump, direct_grid(prob), prob.param, t)
    ifftshift!(pump_buffer2, pump_buffer1)

    if !isnothing(prob.pump)
        @. u[1] += pump_buffer2 * dt
    end

    if !isnothing(prob.noise_prototype)
        randn!(rng, prob.noise_prototype[1])
        grid_map!(pump_buffer, prob.noise_func, u, prob.param)
        @. u[1] += pump_buffer * prob.noise_prototype[1] * √dt
    end
end

_copy!(::ExpFiniteDiff, dest, src) = copy!(dest, src)