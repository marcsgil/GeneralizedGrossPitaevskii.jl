abstract type GGPSolver end

struct StrangSplitting <: GGPSolver end

get_exponential(::AbstractArray{T,N}, ::Nothing, ::NTuple{M}, param, δt) where {T,N,M} = nothing

function get_exponential(u::AbstractArray{T,N}, f!, grid::NTuple{M}, param, δt) where {T,N,M}
    dest = Array{T,N + 1}(undef, size(u, 1), size(u)...)

    function im_f!(dest, x, param)
        f!(dest, x, param)
        lmul!(-im * δt, dest)
    end

    grid_map!(dest, im_f!, grid, param)
    for slice ∈ eachslice(dest, dims=ntuple(n -> n + 2, length(grid)))
        exponential!(slice)
    end

    to_device(u, dest)
end

function get_exponential(u::AbstractArray{T,N}, f, grid::NTuple{N}, param, δt) where {T,N}
    dest = similar(u)
    exp_im_f(x, param) = cis(-δt * f(x, param))
    grid_map!(dest, exp_im_f, grid, param)
    dest
end

mul_or_nothing(::Nothing, δt) = nothing
mul_or_nothing!(::Nothing, δt) = nothing
mul_or_nothing(x, δt) = x * δt
mul_or_nothing!(x, δt) = rmul!(x, δt)


@kernel muladd_kernel!(dest, ::Nothing, ::Nothing) = nothing

@kernel function muladd_kernel!(dest, ::Nothing, drive)
    K = @index(Global, NTuple)
    dest[K...] += drive[K...]
end

@kernel function muladd_kernel!(dest, A, ::Nothing)
    i, K... = @index(Global, NTuple)

    tmp = zero(eltype(dest))
    for j ∈ axes(dest, 1)
        tmp += A[i, j, K...] * dest[j, K...]
    end

    dest[i, K...] = tmp
end

@kernel function muladd_kernel!(dest::AbstractArray{T1,N}, A::AbstractArray{T2,N},
    ::Nothing) where {T1,T2,N}
    K = @index(Global, NTuple)
    dest[K...] *= A[K...]
end

@kernel function muladd_kernel!(dest, A, b)
    i, K... = @index(Global, NTuple)

    tmp = zero(eltype(dest))
    for j ∈ axes(dest, 1)
        tmp += A[i, j, K...] * (dest[j, K...] + b[j, K...])
    end

    dest[i, K...] = tmp
end

@kernel function muladd_kernel!(dest::AbstractArray{T1,N}, A::AbstractArray{T2,N},
    b) where {T1,T2,N}
    K = @index(Global, NTuple)
    dest[K...] += A[K...] * (dest[K...] + b[K...])
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

@kernel function nonlinear_kernel!(ψ, G_δt::Number)
    K = @index(Global, NTuple)
    ψ[K...] *= cis(G_δt * abs2(ψ[K...]))
end

function step!(u, buffer, prob::GrossPitaevskiiProblem, ::StrangSplitting, exp_Aδt, exp_Vδt, G_δt, pump,
    muladd_func!, nonlinear_func!, plan, iplan, t, δt)
    grid_map!(buffer, pump, prob.rs, prob.param, t)
    mul_or_nothing!(buffer, δt / 2)
    muladd_func!(u, exp_Vδt, buffer; ndrange=size(u))
    nonlinear_func!(u, G_δt; ndrange=prob.spatial_size)
    plan * u
    muladd_func!(u, exp_Aδt, nothing; ndrange=size(u))
    iplan * u
    nonlinear_func!(u, G_δt; ndrange=prob.spatial_size)
    muladd_func!(u, exp_Vδt, buffer; ndrange=size(u))
end

similar_or_nothing(x, ::Nothing) = nothing
similar_or_nothing(x, _) = similar(x)

_Progress(::Bool, nsteps) = Progress(nsteps)
_Progress(progress, nsteps) = progress

_next!(progress::Bool, p::Progress) = progress ? next!(p) : nothing

function solve(prob::GrossPitaevskiiProblem, solver::StrangSplitting, nsteps, nsaves, δt,
    t₀=zero(δt); progress=true)
    result = stack(prob.u0 for _ ∈ 1:nsaves+1)
    _result = @view result[ntuple(n -> :, ndims(prob.u0))..., begin+1:end]

    u = similar(prob.u0)
    ifftshift!(u, prob.u0, prob.spatial_dims)
    buffer = similar_or_nothing(u, prob.pump)

    param = prob.param
    exp_Aδt = get_exponential(u, prob.dispersion, prob.ks, param, δt)
    exp_Vδt = get_exponential(u, prob.potential, prob.rs, param, δt)
    G_δt = mul_or_nothing(prob.nonlinearity, δt / 2)

    plan = plan_fft!(u, prob.spatial_dims)
    iplan = inv(plan)

    backend = get_backend(prob.u0)
    muladd_func! = muladd_kernel!(backend)
    nonlinear_func! = nonlinear_kernel!(backend)

    t = t₀

    """return step!(u, buffer, prob, solver, exp_Aδt, exp_Vδt, G_δt, prob.pump!,
                muladd_func!, nonlinear_func!, plan, iplan, t, δt)"""

    """return @benchmark step!($u, $buffer, $prob, $solver, $exp_Aδt, $exp_Vδt, $G_δt, $prob.pump!,
        $muladd_func!, $nonlinear_func!, $plan, $iplan, $t, $δt)"""

    p = _Progress(progress, nsteps)
    for slice ∈ eachslice(_result, dims=ndims(_result))
        for _ ∈ 1:nsteps÷nsaves
            t += δt
            step!(u, buffer, prob, solver, exp_Aδt, exp_Vδt, G_δt, prob.pump,
                muladd_func!, nonlinear_func!, plan, iplan, t, δt)
            _next!(progress, p)
        end
        fftshift!(slice, u)
    end
    finish!(p)

    result
end