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

reshape_or_nothing(::Nothing, size...) = nothing
reshape_or_nothing(f::Function, size...) = (args...) -> reshape([f(args...)], size...)
apply_if_nothing(f, ::Nothing) = nothing
apply_if_nothing(f, x::Number) = x
apply_if_nothing(f, x) = f(x)

"""struct GrossPitaevskiiProblem{T1,T2,T3,T4,T5,T6,T7,T8}
    ψ::T1
    exp_Aδt::T2
    exp_Vδt::T3
    G_δt::T4
    drive::T5
    plan::T6
    iplan::T7
    lengths::T8

    function GrossPitaevskiiProblem(ψ₀, A, V, F, G, δt, lengths)
        spatial_dims = ntuple(x -> x + 1, ndims(ψ₀) - 1)
        @assert length(spatial_dims) == length(lengths)

        plan = plan_fft!(ψ₀, spatial_dims)
        iplan = plan_ifft!(ψ₀, spatial_dims)

        sizes = size(ψ₀)[begin+1:end]

        # Construction of the direct grid
        rs = Iterators.product((fftfreq(n, L) for (n, L) in zip(sizes, lengths))...)

        # Construction of the reciprocal grid
        ks = Iterators.product((fftfreq(n, 2π * n / L) for (n, L) in zip(sizes, lengths))...)

        T = get_unionall(ψ₀)

        # Precompute the exponentials
        exp_Aδt = apply_if_nothing(T, stack(exp(A(k...) * δt) for k ∈ ks))
        exp_Vδt = apply_if_nothing(T, get_exp_Vδt(V, rs, δt / 2))
        drive = apply_if_nothing(T, get_drive(F, A, rs, ks, δt, spatial_dims))
        G_δt = apply_if_nothing(T, get_Gδt(G, δt / 2))

        args = (ifftshift(ψ₀, spatial_dims), exp_Aδt, exp_Vδt, G_δt, drive, plan, iplan, lengths)

        new{typeof.(args)...}(args...)
    end

    function GrossPitaevskiiProblem(ψ₀::AbstractArray{T1,N}, A, V, F, G, δt, lengths::NTuple{N,T2}) where {T1,T2,N}
        _ψ₀ = reshape(ψ₀, 1, size(ψ₀)...)
        _A = reshape_or_nothing(A, 1, 1)
        _V = reshape_or_nothing(V, 1, 1)
        _F = reshape_or_nothing(F, 1)

        GrossPitaevskiiProblem(_ψ₀, _A, _V, _F, G, δt, lengths)
    end
end"""

spatial_dims(x, ::NTuple{N}) where {N} = ntuple(n -> n - N + ndims(x), N)

struct GrossPitaevskyProblem{isscalar,ndims,T1<:AbstractArray,T2<:AbstractFloat,T3,T4,T5,T6,T7,T8}
    u₀::T1
    u::T1
    lengths::NTuple{ndims,T2}
    plan::T3
    iplan::T4
    exp_Aδt::T5
    exp_Vδt::T6
    G_δt::T7
    drive::T8

    function GrossPitaevskyProblem(u₀, lengths, dispersion, potential, nonlinearity, pump)
        @assert ndims(u₀) + 1 == ndims(lengths)

        u = similar(u₀)

        plan = plan_fft!(u, spatial_dims(u, lengths))
        iplan = inv(plan)

        
    end
end

"""
    Most general scenario:

In place functions:
dispersion!(buffer, u, p)
potential!(buffer, u, p)
nonlinearity: constant matrix
pump!()
"""