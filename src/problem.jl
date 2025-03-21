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
    function GrossPitaevskiiProblem(u0, lengths; dispersion::T3=nothing, potential::T4=nothing,
        nonlinearity::T5=nothing, pump::T6=nothing, noise_func::T7=nothing, noise_prototype::T8=nothing,
        param::T9=nothing) where {T3,T4,T5,T6,T7,T8,T9}
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

function direct_grid(prob::GrossPitaevskiiProblem{N}) where {N}
    ntuple(n -> fftfreq(size(first(prob.u0), n), prob.lengths[n]), N)
end

function reciprocal_grid(prob::GrossPitaevskiiProblem{N}) where {N}
    ntuple(n -> fftfreq(size(first(prob.u0), n),
            oftype(prob.lengths[n], 2π) * size(first(prob.u0), n) / prob.lengths[n]), N)
end