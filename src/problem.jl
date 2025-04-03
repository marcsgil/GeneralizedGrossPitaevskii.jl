struct GrossPitaevskiiProblem{N,M,T1,T2,T3,T4,T5,T6,T7,T8,T9}
    u0::NTuple{M,T1}
    lengths::NTuple{N,T2}
    dispersion::T3
    potential::T4
    nonlinearity::T5
    pump::T6
    noise_func::T7
    noise_prototype::T8
    param::T9
    function GrossPitaevskiiProblem(u0::Tuple, lengths::Tuple; dispersion::T3=AdditiveIdentity(), potential::T4=AdditiveIdentity(),
        nonlinearity::T5=AdditiveIdentity(), pump::T6=AdditiveIdentity(), noise_func::T7=AdditiveIdentity(), noise_prototype::T8=AdditiveIdentity(),
        param::T9=AdditiveIdentity()) where {T3,T4,T5,T6,T7,T8,T9}
        @assert all(x -> ndims(x) ≥ length(lengths), u0)
        @assert all(x -> size(x) == size(first(u0)), u0)
        _u0 = complex.(u0)
        _lengths = promote(lengths...)
        N = length(_lengths)
        M = length(_u0)
        T1 = eltype(_u0)
        T2 = typeof(first(_lengths))
        new{N,M,T1,T2,T3,T4,T5,T6,T7,T8,T9}(_u0, _lengths, dispersion, potential, nonlinearity,
            pump, noise_func, noise_prototype, param)
    end
end

function Base.show(io::IO, ::GrossPitaevskiiProblem{N}) where {N}
    print(io, "$(N)D GrossPitaevskiiProblem")
end

direct_grid(L, N) = StepRangeLen(zero(L), L / N, N)

function direct_grid(prob::GrossPitaevskiiProblem)
    map(direct_grid, prob.lengths, size(first(prob.u0)))
end

reciprocal_grid(L, N) = fftfreq(N, oftype(L, 2π) * N / L)

function reciprocal_grid(prob::GrossPitaevskiiProblem{N}) where {N}
    map(reciprocal_grid, prob.lengths, size(first(prob.u0)))
end