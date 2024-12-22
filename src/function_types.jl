struct TensorFunction{T,N}
    f::T
    TensorFunction(f, N) = new{typeof(f),N}(f)
end

(Function::TensorFunction)(args...; kwargs...) = Function.f(args...; kwargs...)

const ScalarFunction{T} = TensorFunction{T,0}
ScalarFunction(f) = TensorFunction(f, 0)
const VectorFunction{T} = TensorFunction{T,1}
VectorFunction(f) = TensorFunction(f, 1)
const MatrixFunction{T} = TensorFunction{T,2}
MatrixFunction(f) = TensorFunction(f, 2)

const UpToVectorFunction = Union{Nothing,ScalarFunction,VectorFunction}
const UpToMatrixFunction = Union{UpToVectorFunction,MatrixFunction}

Base.show(io::IO, F::TensorFunction{T,N}) where {T,N} = print(io, "TensorFunction{$(nameof(F.f)),N}")
Base.show(io::IO, F::ScalarFunction{T}) where {T} = print(io, "ScalarFunction{$(nameof(F.f))}")
Base.show(io::IO, F::VectorFunction{T}) where {T} = print(io, "VectorFunction{$(nameof(F.f))}")
Base.show(io::IO, F::MatrixFunction{T}) where {T} = print(io, "MatrixFunction{$(nameof(F.f))}")