abstract type AbstractGrossPitaevskiiProblem{M,N,T<:Complex,isscalar,T1<:AbstractArray{T,N},T2<:Real,
    FunctionOrNothingT3<:FunctionOrNothing,T4<:FunctionOrNothing,T5,T6<:FunctionOrNothing,T7<:FunctionOrNothing,T8} end

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

convert2scalar(::Nothing) = nothing
convert2scalar(f::Function) = ScalarFunction(f)
convert2scalar(f::ScalarFunction) = f
convert2scalar(::Any) = throw(ArgumentError("A non scalar function is not allowed for a scalar problem."))

function test_signature(::Nothing, u0, lengths, param)
    nothing
end

function test_signature(f::ScalarFunction, u0, lengths, param)
    try
        x = f(lengths, param)
        @assert x isa Number
    catch e
        @warn "A provided function is invalid."
        throw(e)
    end
end

test_signature(f::Function, u0, lengths, param) = test_signature(ScalarFunction(f), u0, lengths, param)

function test_signature(f::MatrixFunction, u0, lengths, param)
    try
        buffer = similar(u0, size(u0,1), size(u0, 1))
        f(buffer, lengths, param)
    catch e
        @warn "A provided function is invalid."
        throw(e)
    end
end

struct GrossPitaevskiiProblem{M,N,T<:Complex,isscalar,
    T1<:AbstractArray{T,N},T2<:Real,T3,T4,T5,T6,T7,T8} <: AbstractGrossPitaevskiiProblem{M,N,T,isscalar,T1,T2,T3,T4,T5,T6,T7,T8}
    u0::T1
    lengths::NTuple{M,T2}
    dispersion::T3
    potential::T4
    nonlinearity::T5
    pump::T6
    noise::T7
    param::T8
    function GrossPitaevskiiProblem(u0, lengths; dispersion::T3=nothing, potential::T4=nothing,
        nonlinearity::T5=nothing, pump::T6=nothing, noise::T7=nothing, param::T8=nothing) where {T3,T4,T5,T6,T7,T8}
        _u0 = complex(u0)
        _lengths = promote(lengths...)
        M = length(lengths)
        N = ndims(_u0)
        T = eltype(_u0)
        @assert N - M ∈ (0, 1) "The dimensions of the initial condition and the lengths do not match."

        for f ∈ (dispersion, potential, pump, noise)
            test_signature(f, _u0, _lengths, param)
        end

        isscalar = Val{M == N}
        T1 = typeof(_u0)
        T2 = typeof(first(_lengths))
        if M == N
            _dispersion = convert2scalar(dispersion)
            _potential = convert2scalar(potential)
            _pump = convert2scalar(pump)
            @assert nonlinearity isa Union{Number,Nothing} "The nonlinearity must be a number for scalar problems."
            _noise = convert2scalar(noise)
            _T3 = typeof(_dispersion)
            _T4 = typeof(_potential)
            _T6 = typeof(_pump)
            _T7 = typeof(_noise)
            new{M,N,T,isscalar,T1,T2,_T3,_T4,T5,_T6,_T7,T8}(_u0, _lengths, _dispersion, _potential, nonlinearity, _pump, _noise, param)
        else
            new{M,N,T,isscalar,T1,T2,T3,T4,T5,T6,T7,T8}(_u0, _lengths, dispersion, potential, nonlinearity, pump, noise, param)
        end
    end
end

function Base.show(io::IO,
    ::GrossPitaevskiiProblem{M,N,T,isscalar,T1}) where {M,N,T,isscalar,T1}
    print(io, "GrossPitaevskiiProblem{$M,$N,$T,$isscalar,$(get_unionall(T1))}")
end

sdims(::GrossPitaevskiiProblem{M,N}) where {M,N} = ntuple(m -> m - M + N, M)