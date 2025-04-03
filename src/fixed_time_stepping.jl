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

    ts, steps_per_save, _dt
end

_Progress(::Nothing, n; kwargs...) = Progress(n; kwargs...)
_Progress(p, n; kwargs...) = p

_next!(::Nothing) = nothing
_next!(p) = next!(p)

_finish!(p, ::Nothing) = finish!(p)
_finish!(p, progress) = nothing

function solve(prob, alg::FixedTimeSteppingAlgorithm, tspan;
    dt,
    nsaves,
    show_progress=true,
    progress=nothing,
    save_start=true,
    fftw_num_threads=1,
    workgroup_size=(),
    rng=Random.default_rng()
)
    FFTW.set_num_threads(fftw_num_threads)

    ts, steps_per_save, _dt = resolve_fixed_timestepping(dt, tspan, nsaves)

    result, u, args... = get_precomputations(alg, prob, _dt, tspan, nsaves, workgroup_size, save_start)
    _result = map(result) do x
        @view x[ntuple(n -> :, ndims(first(result)) - 1)..., begin+save_start:end]
    end

    t = tspan[1]
    p = _Progress(progress, steps_per_save * nsaves; enabled=show_progress)
    for n ∈ axes(first(_result), ndims(first(_result)))
        slice = map(_result) do x
            @view x[ntuple(n -> :, ndims(first(result)) - 1)..., n]
        end

        for _ ∈ 1:steps_per_save
            t += _dt
            step!(alg, t, _dt, u, prob, rng, args...)
            _next!(p)
        end
        map(copy!, slice, u)
        ts[n+1] = t
    end
    _finish!(p, progress)

    ts[begin+1-save_start:end], result
end