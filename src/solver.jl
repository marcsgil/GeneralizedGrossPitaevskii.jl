@kernel muladd_kernel!(ψ, ::Nothing, ::Nothing) = nothing

@kernel function muladd_kernel!(ψ, A, ::Nothing)
    i, K... = @index(Global, NTuple)

    tmp = zero(eltype(ψ))
    for j ∈ axes(ψ, 1)
        tmp += A[i, j, K...] * ψ[j, K...]
    end

    ψ[i, K...] = tmp
end

@kernel function muladd_kernel!(ψ, A, b)
    i, K... = @index(Global, NTuple)

    tmp = zero(eltype(ψ))
    for j ∈ axes(ψ, 1)
        tmp += A[i, j, K...] * ψ[j, K...]
    end

    ψ[i, K...] = tmp + b[i, K...]
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

function strang_splitting_step!(ψ, exp_Aδt, exp_Vδt, drive, G_δt, ndrange,
    muladd_func!, nonlinear_func!, plan, iplan)
    muladd_func!(ψ, exp_Vδt, nothing; ndrange)
    nonlinear_func!(ψ, G_δt; ndrange=ndrange[begin+1:end])
    plan * ψ
    muladd_func!(ψ, exp_Aδt, drive; ndrange)
    iplan * ψ
    nonlinear_func!(ψ, G_δt; ndrange=ndrange[begin+1:end])
    muladd_func!(ψ, exp_Vδt, nothing; ndrange)
end

function strang_splitting_step!(prob::GrossPitaevskiiProblem, muladd_func!, nonlinear_func!)
    strang_splitting_step!(prob.ψ, prob.exp_Aδt, prob.exp_Vδt, prob.drive, prob.G_δt, size(prob.ψ),
        muladd_func!, nonlinear_func!, prob.plan, prob.iplan)
end

function solve(prob::GrossPitaevskiiProblem, nsteps, save_every;
    progress=nothing)

    # Get kernel functions
    backend = get_backend(prob.ψ)
    muladd_func! = muladd_kernel!(backend)
    nonlinear_func! = nonlinear_kernel!(backend)
    spatial_dims = ntuple(x -> x + 1, ndims(prob.ψ) - 1)

    # Initialize the result array
    result = similar(prob.ψ, size(prob.ψ)..., nsteps ÷ save_every + 1)
    fftshift!(view(result, :, :, :, 1), prob.ψ, spatial_dims)

    # Start the time evolution
    for n ∈ 1:nsteps
        # Perform a Strang splitting step
        strang_splitting_step!(prob, muladd_func!, nonlinear_func!)

        # Save the result if necessary
        if n % save_every == 0
            fftshift!(view(result, :, :, :, n ÷ save_every + 1), prob.ψ, spatial_dims)
        end

        # Update the progress bar
        if !isnothing(progress)
            next!(progress)
        end
    end
    result
end