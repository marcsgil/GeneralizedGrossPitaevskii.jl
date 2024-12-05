abstract type GGPSolver end

struct StrangSplitting <: GGPSolver end

get_exponential(::Nothing, u, rs, δt; param) = nothing

function get_exponential(f!, u, grid, δt; param)
    dest = Array{eltype(u),ndims(u) + 1}(undef, size(u, 1), size(u)...)
    T = get_unionall(u)

    function im_f!(dest, x...; param)
        f!(dest, x...; param)
        lmul!(-im * δt, dest)
    end

    grid_map!(dest, im_f!, grid...; param)
    for slice ∈ eachslice(dest, dims=ntuple(n -> n + 2, length(grid)))
        exponential!(slice)
    end

    T(dest)
end

mul_or_nothing(::Nothing, δt) = nothing
mul_or_nothing(x, δt) = x * δt
mul_or_nothing!(x, δt) = lmul!(δt, x)
mul_or_nothing!(::Nothing, δt) = nothing

@kernel matmul_slices_kernel!(dest, ::Nothing, ::Nothing) = nothing

@kernel function matmul_slices_kernel!(ψ, ::Nothing, drive)
    K = @index(Global, NTuple)
    ψ[K...] += drive[K...]
end

@kernel function matmul_slices_kernel!(ψ, A, ::Nothing)
    i, K... = @index(Global, NTuple)

    tmp = zero(eltype(ψ))
    for j ∈ axes(ψ, 1)
        tmp += A[i, j, K...] * ψ[j, K...]
    end

    ψ[i, K...] = tmp
end

@kernel function matmul_slices_kernel!(ψ, A, b)
    i, K... = @index(Global, NTuple)

    tmp = zero(eltype(ψ))
    for j ∈ axes(ψ, 1)
        tmp += A[i, j, K...] * (ψ[j, K...] + b[j, K...])
    end

    ψ[i, K...] = tmp
end

@kernel nonlinear_kernel!(ψ, ::Nothing) = nothing

@kernel function nonlinear_kernel!(ψ, G_δt)
    K = @index(Global, NTuple)

    tmp = zero(eltype(ψ))
    for n ∈ axes(G_δt, 2), m ∈ axes(G_δt, 1)
        tmp += G_δt[m, n] * conj(ψ[m, K...]) * ψ[n, K...]
    end

    for i ∈ axes(ψ, 1)
        ψ[i, K...] *= cis(tmp)
    end
end

function step!(u, buffer, prob::GrossPitaevskiiProblem, ::StrangSplitting, exp_Aδt, exp_Vδt, G_δt, pump!,
    matmul_slices_func!, nonlinear_func!, plan, iplan, t, δt)
    grid_map!(buffer, pump!, prob.rs...; param=(t, prob.param...))
    mul_or_nothing!(buffer, δt)
    matmul_slices_func!(u, exp_Vδt, buffer; ndrange=size(u))
    nonlinear_func!(u, G_δt; ndrange=prob.spatial_size)
    plan * u
    matmul_slices_func!(u, exp_Aδt, nothing; ndrange=size(u))
    iplan * u
    nonlinear_func!(u, G_δt; ndrange=prob.spatial_size)
    matmul_slices_func!(u, exp_Vδt, buffer; ndrange=size(u))
end

similar_or_nothing(x, ::Nothing) = nothing
similar_or_nothing(x, _) = similar(x)

function solve(prob::GrossPitaevskiiProblem, solver::StrangSplitting, nsteps, nsaves, δt, t₀=zero(δt))
    result = stack(prob.u₀ for _ ∈ 1:nsaves+1)
    _result = @view result[ntuple(n -> :, ndims(prob.u₀))..., begin+1:end]

    u = similar(prob.u₀)
    ifftshift!(u, prob.u₀, prob.spatial_dims)
    buffer = similar_or_nothing(u, prob.pump!)

    param = prob.param
    exp_Aδt = get_exponential(prob.dispersion!, u, prob.ks, δt; param)
    exp_Vδt = get_exponential(prob.potential!, u, prob.rs, δt / 2; param)
    G_δt = mul_or_nothing(prob.nonlinearity, δt / 2)

    plan = plan_fft!(u, prob.spatial_dims)
    iplan = inv(plan)

    backend = get_backend(prob.u₀)
    matmul_slices_func! = matmul_slices_kernel!(backend)
    nonlinear_func! = nonlinear_kernel!(backend)

    t = t₀

    #return @benchmark step!($u, $buffer, $prob, $solver, $exp_Aδt, $exp_Vδt, $G_δt, $prob.pump!,
    #$matmul_slices_func!, $nonlinear_func!, $plan, $iplan, $t, $δt)

    for slice ∈ eachslice(_result, dims=ndims(_result))
        for _ ∈ 1:nsteps÷nsaves
            t += δt
            step!(u, buffer, prob, solver, exp_Aδt, exp_Vδt, G_δt, prob.pump!,
                matmul_slices_func!, nonlinear_func!, plan, iplan, t, δt)
        end
        fftshift!(slice, u)
    end
    
    result
end