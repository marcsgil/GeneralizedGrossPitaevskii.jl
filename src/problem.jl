abstract type AbstractGrossPitaevskiiProblem{M,N,T<:Complex,isscalar,T1<:AbstractArray{T,N},T2<:Real,
    T3<:UpToMatrixFunction,T4<:UpToMatrixFunction,T5<:Union{Nothing,Number,AbstractVecOrMat},
    T6<:UpToVectorFunction,T7<:UpToMatrixFunction,T8} end

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

build_buffer(f, u0) = nothing
build_buffer(::TensorFunction{T,N}, u0) where {T,N} = similar(u0, ntuple(n->size(u0, 1), N)...)

call_func(::Nothing, buffer, lengths, param) = nothing
function call_func(f::ScalarFunction, buffer, lengths, param)
    x = f(lengths, param)
    @assert x isa Number
end
call_func(f::Union{VectorFunction,MatrixFunction}, buffer, lengths, param) = f(buffer, lengths, param)

function test_signature(f::UpToMatrixFunction, u0, lengths, param, name)
    buffer = build_buffer(f, u0)
    try
        call_func(f, buffer, lengths, param)
    catch e
        throw(ArgumentError("""
        There is an error in the problem specification, identified when calling $name.
        Check if the signatures, return types and parameters are correct.
        The stacktrace bellow may help you to find the error:

        $e
        """))
    end
end

function test_signature(f::Function, u0, lengths, param, name)
    throw(ArgumentError("The provided $name is a Function. You should wrap it in a `ScalarFunction`, `VectorFunction` or `MatrixFunction`."))
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

        func_dict = Dict(
            "dispersion" => dispersion,
            "potential" => potential,
            "noise" => noise
        )
        for (name, f) ∈ func_dict
            test_signature(f, _u0, _lengths, param, name)
        end

        isscalar = Val{M == N}
        T1 = typeof(_u0)
        T2 = typeof(first(_lengths))
        new{M,N,T,isscalar,T1,T2,T3,T4,T5,T6,T7,T8}(_u0, _lengths, dispersion, potential, nonlinearity, pump, noise, param)
    end
end

function Base.show(io::IO,
    ::GrossPitaevskiiProblem{M,N,T,isscalar,T1}) where {M,N,T,isscalar,T1}
    print(io, "GrossPitaevskiiProblem{$M,$N,$T,$isscalar,$(get_unionall(T1))}")
end

sdims(::GrossPitaevskiiProblem{M,N}) where {M,N} = ntuple(m -> m - M + N, M)