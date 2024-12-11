abstract type AbstractGrossPitaevskiiProblem{M,N,T,T1<:AbstractArray{T,N},T2<:Real,T3,T4,T5,T6,T7} end

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

convert2complex(x::AbstractArray{T}) where {T} = complex(float(x))
convert2complex(x::AbstractArray{T}) where {T<:AbstractFloat} = complex(x)

struct GrossPitaevskiiProblem{M,N,T<:Complex,
    T1<:AbstractArray{T,N},T2<:Real,T3,T4,T5,T6,T7} <: AbstractGrossPitaevskiiProblem{M,N,T,T1,T2,T3,T4,T5,T6,T7}
    u0::T1
    lengths::NTuple{M,T2}
    dispersion::T3
    potential::T4
    nonlinearity::T5
    pump::T6
    param::T7
end

GrossPitaevskiiProblem(u0::AbstractArray{T,N}, lengths,
    dispersion, potential, nonlinearity, pump, param=nothing) where {T<:Complex,N} =
    GrossPitaevskiiProblem(u0, lengths, dispersion, potential, nonlinearity, pump, param)

function Base.show(io::IO,
    ::GrossPitaevskiiProblem{M,N,T,T1}) where {M,N,T,T1}
    print(io, "GrossPitaevskiiProblem{$M,$N,$T,$(get_unionall(T1))}")
end

sdims(::GrossPitaevskiiProblem{M,N}) where {M,N} = ntuple(m -> m - M + N, M)


function evaluate_pump!(::AbstractGrossPitaevskiiProblem{M,N,T,T1,T2,T3,T4,T5,Nothing,T7},
    args...) where {M,N,T,T1,T2,T3,T4,T5,T7}
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