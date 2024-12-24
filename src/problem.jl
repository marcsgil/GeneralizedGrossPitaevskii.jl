struct GrossPitaevskiiProblem{N,T,T1<:AbstractArray{T,N},T2,T3,T4,T5,T6,T7,T8}
    u0::T1
    lengths::NTuple{N,T2}
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
        N = ndims(_u0)
        T = eltype(_u0)
        T1 = typeof(_u0)
        T2 = typeof(first(_lengths))
        new{N,T,T1,T2,T3,T4,T5,T6,T7,T8}(_u0, _lengths, dispersion, potential, nonlinearity, pump, noise, param)
    end
end

function Base.show(io::IO,
    ::GrossPitaevskiiProblem{N,T,T1}) where {N,T,T1}
    print(io, "GrossPitaevskiiProblem{$T1}")
end

Base.size(prob::GrossPitaevskiiProblem, args...) = size(prob.u0, args...)
Base.ndims(prob::GrossPitaevskiiProblem{N}) where {N} = ndims(prob.u0)
Base.eltype(::GrossPitaevskiiProblem{N,T}) where {N,T} = T

"""
    nsdims(prob)

Return the number of spatial dimensions of the problem.
"""
nsdims(::GrossPitaevskiiProblem{N}) where {N} = N

"""
    ssize(prob[, dim])

Return a tuple representing the spatial size of the problem.
If `dim` is specified, return the size of the `dim`-th spatial dimension.
"""
ssize(prob::GrossPitaevskiiProblem, args...) = size(prob, args...)

"""
    sdims(prob)

Return the spatial dimensions of the problem.
"""
sdims(::GrossPitaevskiiProblem{N}) where {N} = ntuple(identity, N)

function direct_grid(prob::GrossPitaevskiiProblem{N}) where {N}
    ntuple(n -> fftfreq(ssize(prob, n), prob.lengths[n]), N)
end

function reciprocal_grid(prob::GrossPitaevskiiProblem{N}) where {N}
    ntuple(n -> fftfreq(ssize(prob, n), 2π * ssize(prob, n) / prob.lengths[n]), N)
end

build_buffer(f, u0) = nothing
build_buffer(::TensorFunction{T,N}, u0) where {T,N} = similar(u0, ntuple(n -> size(u0, 1), N)...)

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