abstract type FixedTimeSteppingAlgorithm end

abstract type FixedTimeSteppingIterator end

"""
    resolve_fixed_timestepping(dt, tspan, nsaves)

This function resolves the fixed timestepping for the given solver and time span.
The problem is that the number of saves, time span and time step will, in general, not be compatible.
This functions resolves this by calculating a smaller time step `_dt` such that everything fits.
It returns a Tuple containing a vector of time points `ts` (only the first entry is initialized `ts[1] = tspan[1]`), 
the number of steps per save `steps_per_save` and the new time step `_dt`.
"""
function resolve_fixed_timestepping(dt, tspan, nsaves)
    T = float(promote_type(typeof(dt), typeof(first(tspan)), typeof(last(tspan))))
    ts = Vector{T}(undef, nsaves + 1)
    ts[1] = first(tspan)

    ΔT = T(last(tspan) - first(tspan)) / nsaves
    steps_per_save = round(Int, ΔT / dt, RoundUp)
    _dt = ΔT / steps_per_save

    _dt, ts, steps_per_save
end

function solve!(iter::FixedTimeSteppingIterator)
    save_start = iter.save_start
    steps_per_save = iter.steps_per_save
    dt = iter.dt
    p = iter.progress
    ts = iter.ts

    result = map(iter.result) do x
        @view x[ntuple(n -> :, ndims(first(iter.result)) - 1)..., begin+save_start:end]
    end

    t = ts[1]
    for n ∈ axes(first(result), ndims(first(result)))
        slice = map(result) do x
            @view x[ntuple(n -> :, ndims(first(result)) - 1)..., n]
        end

        for _ ∈ 1:steps_per_save
            t += dt
            step!(iter, t, dt)
            _next!(p)
        end
        map(copy!, slice, iter.u)
        ts[n+1] = t
    end
    _finish!(p, iter.given_progress)

    ts[begin+1-save_start:end], iter.result
end

"""
    solve(prob, alg, tspan;
    dt,
    nsaves,
    show_progress=true,
    progress=nothing,
    save_start=true,
    workgroup_size=(),
    rng=nothing)

This function solves the given `prob` using the algorithm `alg` over the time span `tspan`.

It uses a fixed time step `dt` and saves the solution `nsaves` times during the simulation.

The `show_progress` argument controls whether a progress bar is shown, and `progress` can be used to provide a custom progress bar from `ProgressMeter.jl`.

The `save_start` argument indicates whether the initial condition should be saved. 
If `true`, the initial condition is saved as the first entry in the result, and the size of the time dimension of the result is `nsaves + save_start`.

The `workgroup_size` argument can be used to specify the workgroup size for the kernel functions.
    
The `rng` argument can be used to provide a random number generator for the simulation.
"""
function solve(prob::GrossPitaevskiiProblem, args...; kwargs...)
    solve!(init(prob, args...; kwargs...))
end