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

get_drive(::Nothing, As, rs, ks, δt, spatial_dims) = nothing

function get_drive(F, A, rs, ks, δt, spatial_dims)
    Fs = stack(F(r...) for r ∈ rs)
    F̃ = fft(Fs, spatial_dims)

    stack(map(eachslice(F̃, dims=spatial_dims), ks) do _F, k
        _A = A(k...)
        _A \ (exp(_A * δt) * _F - _F)
    end)
end

get_exp_Vδt(::Nothing, rs, δt) = nothing
get_exp_Vδt(V, rs, δt) = stack(exp(V(r...) * δt) for r ∈ rs)

get_Gδt(::Nothing, δt) = nothing
get_Gδt(G, δt) = G * δt

function solve(ψ₀, A, V, F, G,
    δt, nsteps, save_every, lengths;
    progress=nothing)

    spatial_dims = ntuple(x -> x + 1, ndims(ψ₀) - 1)
    @assert length(spatial_dims) == length(lengths)
    sizes = size(ψ₀)[begin+1:end]

    # Construction of the direct grid
    rs = Iterators.product((fftfreq(n, L) for (n, L) in zip(sizes, lengths))...)

    # Construction of the reciprocal grid
    ks = Iterators.product((fftfreq(n, 2π * n / L) for (n, L) in zip(sizes, lengths))...)

    # Precompute the exponentials
    exp_Aδt = stack(exp(A(k...) * δt) for k ∈ ks)

    exp_Vδt = get_exp_Vδt(V, rs, δt / 2)
    drive = get_drive(F, A, rs, ks, δt, spatial_dims)
    G_δt = get_Gδt(G, δt / 2)

    # Precompute the sizes
    ndrange = size(ψ₀)

    # Get kernel functions
    backend = get_backend(ψ₀)
    muladd_func! = muladd_kernel!(backend)
    nonlinear_func! = nonlinear_kernel!(backend)

    # Precompute the FFT plans
    plan = plan_fft!(ψ₀)
    iplan = plan_ifft!(ψ₀)

    # Initialize the result array
    result = stack(ψ₀ for _ ∈ 1:nsteps÷save_every+1)

    # Initialize the shifted field function
    ψ = ifftshift(ψ₀, spatial_dims)

    # Start the time evolution
    for n ∈ 1:nsteps
        # Perform a Strang splitting step
        strang_splitting_step!(ψ, exp_Aδt, exp_Vδt, drive, G_δt, ndrange,
            muladd_func!, nonlinear_func!, plan, iplan)

        # Save the result if necessary
        if n % save_every == 0
            fftshift!(view(result, :, :, :, n ÷ save_every + 1), ψ, spatial_dims)
        end

        # Update the progress bar
        if !isnothing(progress)
            next!(progress)
        end
    end
    result
end

reshape_or_nothing(::Nothing, size...) = nothing
reshape_or_nothing(f::Function, size...) = (args...) -> reshape([f(args...)], size...)

function solve(ψ₀::AbstractArray{T1,N}, A, V, F, G,
    δt, nsteps, save_every, lengths::NTuple{N,T2};
    progress=nothing) where {T1,T2,N}

    _ψ₀ = reshape(ψ₀, 1, size(ψ₀)...)
    _A = reshape_or_nothing(A, 1, 1)
    _V = reshape_or_nothing(V, 1, 1)
    _F = reshape_or_nothing(F, 1)

    result = solve(_ψ₀, _A, _V, _F, G, δt, nsteps, save_every, lengths; progress=progress)

    dropdims(result, dims=1)
end