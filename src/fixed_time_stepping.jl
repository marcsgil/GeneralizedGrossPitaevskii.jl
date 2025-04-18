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
    ts = Vector{typeof(dt)}(undef, nsaves + 1)
    ts[1] = tspan[1]

    ΔT = (tspan[2] - tspan[1]) / nsaves
    steps_per_save = round(Int, ΔT / dt, RoundUp)
    _dt = ΔT / steps_per_save

    _dt, ts, steps_per_save
end

function CommonSolve.solve!(iter::FixedTimeSteppingIterator)
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
            CommonSolve.step!(iter, t, dt)
            _next!(p)
        end
        map(copy!, slice, iter.u)
        ts[n+1] = t
    end
    #_finish!(p, progress)

    ts[begin+1-save_start:end], iter.result
end