abstract type AbstractGrossPitaevskiiProblem{M,N,T,isscalar,T1<:AbstractArray{T,N},T2<:Real,T3,T4,T5,T6,T7} end

Base.size(prob::AbstractGrossPitaevskiiProblem, args...) = size(prob.u0, args...)
Base.ndims(::AbstractGrossPitaevskiiProblem{M,N}) where {M,N} = N
Base.eltype(::AbstractGrossPitaevskiiProblem{M,N,T}) where {M,N,T} = T

"""
    nsdims(prob)

Return the number of spatial dimensions of the problem.
"""
nsdims(::AbstractGrossPitaevskiiProblem{M}) where {M} = M

"""
    sdims(prob)

Return the spatial dimensions of the problem.
"""
function sdims end

"""
    ssize(prob[, dim])

Return a tuple representing the spatial size of the problem.
If `dim` is specified, return the size of the `dim`-th spatial dimension.
"""
function ssize(prob::AbstractGrossPitaevskiiProblem{M}) where {M}
    J = sdims(prob)
    ntuple(m -> size(prob.u0, J[m]), M)
end
ssize(prob, dim) = ssize(prob)[dim]

function direct_grid(prob::AbstractGrossPitaevskiiProblem{M}) where {M}
    ntuple(m -> fftfreq(ssize(prob, m), prob.lengths[m]), M)
end

function reciprocal_grid(prob::AbstractGrossPitaevskiiProblem{M}) where {M}
    ntuple(m -> fftfreq(ssize(prob, m), 2π * ssize(prob, m) / prob.lengths[m]), M)
end

function evaluate_pump!(::AbstractGrossPitaevskiiProblem{M,N,T,isscalar,T1,T2,T3,T4,T5,Nothing,T7},
    args...) where {M,N,T,isscalar,T1,T2,T3,T4,T5,T7}
    nothing
end

function evaluate_pump!(prob, dest, t)
    rs = direct_grid(prob)
    grid_map!(dest, prob.pump, rs, prob.param, t)
end

function evaluate_pump!(prob, dest_next, dest_now, t)
    copy!(dest_now, dest_next)
    evaluate_pump!(prob, dest_next, t)
end

struct GrossPitaevskiiProblem{M,N,T<:Complex,isscalar<:Union{Val{true},Val{false}},
    T1<:AbstractArray{T,N},T2<:Real,T3,T4,T5,T6,T7} <: AbstractGrossPitaevskiiProblem{M,N,T,isscalar,T1,T2,T3,T4,T5,T6,T7}
    u0::T1
    lengths::NTuple{M,T2}
    dispersion::T3
    potential::T4
    nonlinearity::T5
    pump::T6
    param::T7
    function GrossPitaevskiiProblem(u0, lengths, dispersion::T3, potential::T4, nonlinearity::T5, pump::T6,
        param::T7=nothing) where {T3,T4,T5,T6,T7}
        _u0 = complex(u0)
        _lengths = promote(lengths...)
        M = length(lengths)
        N = ndims(_u0)
        T = eltype(_u0)
        @assert N - M ∈ (0, 1)
        isscalar = Val{M == N}
        T1 = typeof(_u0)
        T2 = typeof(first(_lengths))
        new{M,N,T,isscalar,T1,T2,T3,T4,T5,T6,T7}(_u0, _lengths, dispersion, potential, nonlinearity, pump, param)
    end
end

function Base.show(io::IO,
    ::GrossPitaevskiiProblem{M,N,T,isscalar,T1}) where {M,N,T,isscalar,T1}
    print(io, "GrossPitaevskiiProblem{$M,$N,$T,$isscalar,$(get_unionall(T1))}")
end

sdims(::GrossPitaevskiiProblem{M,N}) where {M,N} = ntuple(m -> m - M + N, M)

struct StochasticGrossPitaevskiiProblem{M,N,T<:Complex,isscalar<:Union{Val{true},Val{false}},
    T1<:AbstractArray{T,N},T2<:Real,T3,T4,T5,T6,T7,T8} <: AbstractGrossPitaevskiiProblem{M,N,T,isscalar,T1,T2,T3,T4,T5,T6,T7}
    u0::T1
    lengths::NTuple{M,T2}
    dispersion::T3
    potential::T4
    nonlinearity::T5
    pump::T6
    param::T7
    noise_func::T8
    function StochasticGrossPitaevskiiProblem(u0, lengths, dispersion::T3, potential::T4, nonlinearity::T5, pump::T6, noise_func::T8,
        param::T7=nothing) where {T3,T4,T5,T6,T7,T8}
        _u0 = complex(u0)
        _lengths = promote(lengths...)
        M = length(lengths)
        N = ndims(_u0)
        T = eltype(_u0)
        @assert N - M ∈ (1, 2)
        isscalar = Val{M + 1 == N}
        T1 = typeof(_u0)
        T2 = typeof(first(_lengths))
        new{M,N,T,isscalar,T1,T2,T3,T4,T5,T6,T7,T8}(_u0, _lengths, dispersion, potential, nonlinearity, pump, param, noise_func)
    end
end

function Base.show(io::IO,
    ::StochasticGrossPitaevskiiProblem{M,N,T,isscalar,T1}) where {M,N,T,isscalar,T1}
    print(io, "StochasticGrossPitaevskiiProblem{$M,$N,$T,$isscalar,$(get_unionall(T1))}")
end

sdims(::StochasticGrossPitaevskiiProblem{M,N}) where {M,N} = ntuple(m -> m - M + N - 1, M)