struct GrossPitaevskiiProblem{M,N,T,T1<:AbstractArray{T,M},T2,T3,T4,T5,T6,T7,T8,T9}
    u0::T1
    lengths::NTuple{N,T2}
    dispersion::T3
    potential::T4
    nonlinearity::T5
    pump::T6
    noise_func::T7
    noise_eltype::T8
    param::T9
    function GrossPitaevskiiProblem(u0, lengths; dispersion::T3=nothing, potential::T4=nothing,
        nonlinearity::T5=nothing, pump::T6=nothing, noise_func::T7=nothing, noise_eltype::T8=nothing,
        param::T9=nothing) where {T3,T4,T5,T6,T7,T8,T9}
        @assert ndims(u0) ≥ length(lengths)
        _u0 = complex(u0)
        _lengths = promote(lengths...)
        M = ndims(_u0)
        N = length(_lengths)
        T = eltype(_u0)
        T1 = typeof(_u0)
        T2 = typeof(first(_lengths))
        new{M,N,T,T1,T2,T3,T4,T5,T6,T7,T8,T9}(_u0, _lengths, dispersion, potential, nonlinearity,
            pump, noise_func, noise_eltype, param)
    end
end

function Base.show(io::IO,
    ::GrossPitaevskiiProblem{M,N,T,T1}) where {M,N,T,T1}
    print(io, "$(N)D GrossPitaevskiiProblem{$T1}")
end

function direct_grid(prob::GrossPitaevskiiProblem{M,N}) where {M,N}
    ntuple(n -> fftfreq(size(prob.u0, n), prob.lengths[n]), N)
end

function reciprocal_grid(prob::GrossPitaevskiiProblem{M,N}) where {M,N}
    ntuple(n -> fftfreq(size(prob.u0, n),
            oftype(prob.lengths[n], 2π) * size(prob.u0, n) / prob.lengths[n]), N)
end