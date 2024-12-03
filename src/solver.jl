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

@kernel function matmul_slices_kernel!(ψ, A, b)
    j, K... = @index(Global, NTuple)
    ψ[j, K...] += b[j, K...]
end

@kernel nonlinear_kernel!(ψ, ::Nothing) = nothing

@kernel function nonlinear_kernel!(ψ, G_δt)
    K = @index(Global, NTuple)

    tmp = zero(eltype(ψ))
    for n ∈ axes(G_δt, 2), m ∈ axes(G_δt, 1)
        tmp -= G_δt[m, n] * conj(ψ[m, K...]) * ψ[n, K...]
    end

    for i ∈ axes(ψ, 1)
        ψ[i, K...] *= cis(tmp)
    end
end

function strang_splitting_step!(ψ, exp_Aδt, exp_Vδt, drive, G_δt, ndrange,
    matmul_slices_func!, nonlinear_func!, plan, iplan)
    matmul_slices_func!(ψ, exp_Vδt, drive; ndrange)
    nonlinear_func!(ψ, G_δt; ndrange=ndrange[begin+1:end])
    plan * ψ
    matmul_slices_func!(ψ, exp_Aδt, nothing; ndrange)
    iplan * ψ
    nonlinear_func!(ψ, G_δt; ndrange=ndrange[begin+1:end])
    matmul_slices_func!(ψ, exp_Vδt, drive; ndrange)
end

function strang_splitting_step!(prob::GrossPitaevskiiProblem, matmul_slices_func!, nonlinear_func!, t)
    ssize = spatial_size(prob)
    rs = (fftfreq(n, l) for (n, l) ∈ zip(ssize, prob.lengths))
    grid_map!(prob.buffer, prob.pump!, rs...)
    strang_splitting_step!(prob.u, prob.exp_Aδt, prob.exp_Vδt, prob.buffer, prob.G_δt, size(prob.u),
        matmul_slices_func!, nonlinear_func!, prob.plan, prob.iplan)
end

function solve(prob::GrossPitaevskiiProblem, nsteps, save_every;
    progress=nothing)

    # Get kernel functions
    backend = get_backend(prob.u)
    matmul_slices_func! = matmul_slices_kernel!(backend)
    nonlinear_func! = nonlinear_kernel!(backend)
    sdims = spatial_dims(prob)

    # Initialize the result array
    result = similar(prob.u, size(prob.u)..., nsteps ÷ save_every + 1)
    result[.., 1] = prob.u₀
    fftshift!(prob.u, prob.u₀, sdims)

    # Start the time evolution
    for n ∈ 1:nsteps
        # Perform a Strang splitting step
        t = n * prob.δt
        strang_splitting_step!(prob, matmul_slices_func!, nonlinear_func!, t)

        # Save the result if necessary
        if n % save_every == 0
            fftshift!(view(result, :, :, :, n ÷ save_every + 1), prob.u, sdims)
        end

        # Update the progress bar
        if !isnothing(progress)
            next!(progress)
        end
    end
    result
end