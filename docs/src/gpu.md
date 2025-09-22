# GPU Support

GeneralizedGrossPitaevskii.jl provides seamless GPU acceleration through [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), enabling the same code to run efficiently on both CPU and GPU hardware. The package automatically detects array types and dispatches to appropriate compute backends, making GPU usage as simple as providing GPU arrays as initial conditions.

This page explains how to enable GPU support, configure backends, and optimize performance for GPU simulations.

## [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)

[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) provides a unified interface for writing GPU kernels that work across different backends (CUDA, ROCm, oneAPI, etc.). GeneralizedGrossPitaevskii.jl leverages this abstraction to provide hardware-agnostic simulations. Beyond CPU, the package has only been thested with the CUDA backend. The other backends are untested.

## Specifying the GPU backend

Different GPU backends require different array packages and setup procedures. The choice of backend depends on your hardware and software environment.

For NVIDIA GPUs, use [CUDA.jl](https://cuda.juliagpu.org/stable/), which needs to be installed separately:

```julia
using CUDA, GeneralizedGrossPitaevskii

# Check GPU availability
CUDA.functional()  # Should return true

# Create GPU arrays
u0 = ((CUDA.zeros(ComplexF64, 512, 512)),)

# Standard problem setup - automatically uses GPU
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity, param)
ts, sol = solve(prob, StrangSplitting(), tspan; dt, nsaves)
```

The other backends should work similarly, but have not been tested.

## Performance considerations

### Floating Point Precision

GPU performance often benefits significantly from reduced precision, such as `Float32` or `ComplexF32`, compared to `Float64` or `ComplexF64`.
```julia
# Double precision (slower, higher accuracy)
u0_double = (CuArray(zeros(ComplexF64, N, N)),)

# Single precision (faster, sufficient for most applications)
u0_single = (CuArray(zeros(ComplexF32, N, N)),)
```

Make sure to also change the types in the parameters and functions you provide (e.g., dispersion, nonlinearity) to match the precision of your arrays.

Consider the following factors when choosing precision:
- **Accuracy requirements**: Some applications may require higher precision
- **Memory usage**: Lower precision reduces memory consumption, allowing larger problems

### Performance Tips

1. **Use appropriate precision**: Float32 for most applications, Float64 only when necessary.
2. **Choose optimal grid sizes**: Powers of 2 often perform better for FFTs
3. **Ensemble simulations**: GPU parallelism is ideal for multiple stochastic trajectories. Be sure to use a large enough ensemble size to fully utilize the GPU.
4. **Tune workgroup sizes**: A `workgroup_size` parameter is available in the `solve` function to allow tuning of workgroup sizes for better performance.