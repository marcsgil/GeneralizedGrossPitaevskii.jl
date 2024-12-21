struct TensorFunction{T,N}
    f::T
    TensorFunction(f, N) = new{typeof(f), N}(f)
end

(Function::TensorFunction)(args...; kwargs...) = Function.f(args...; kwargs...)

const ScalarFunction{T} = TensorFunction{T, 0}
ScalarFunction(f) = TensorFunction(f, 0)
const VectorFunction{T} = TensorFunction{T, 1}
VectorFunction(f) = TensorFunction(f, 1)
const MatrixFunction{T} = TensorFunction{T, 2}
MatrixFunction(f) = TensorFunction(f, 2)

const FunctionOrNothing = Union{TensorFunction, Nothing}