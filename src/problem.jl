spatial_dims(x, ::NTuple{N}) where {N} = ntuple(n -> n - N + ndims(x), N)
spatial_size(x, ::NTuple{N}) where {N} = ntuple(n -> size(x, n - N + ndims(x)), N)

get_exponential(::Nothing, u, rs, δt) = nothing

function get_exponential(f!, u, grid, δt)
    dest = Array{eltype(u),ndims(u) + 1}(undef, size(u, 1), size(u)...)
    T = get_unionall(u)

    function im_f!(dest, x...)
        f!(dest, x...)
        lmul!(-im * δt, dest)
    end

    grid_map!(dest, im_f!, grid...)
    for slice ∈ eachslice(dest, dims=(3, 4))
        exponential!(slice)
    end

    T(dest)
end

get_Gδt(::Nothing, δt) = nothing
get_Gδt(nonlinearity, δt) = nonlinearity * δt

struct GrossPitaevskiiProblem{N,T1,T2,T3,T4,T5,T6,T7,T8}
    u₀::T1
    u::T1
    buffer::T1
    δt::T2
    lengths::NTuple{N,T2}
    exp_Aδt::T3
    exp_Vδt::T4
    G_δt::T5
    pump!::T6
    plan::T7
    iplan::T8

    function GrossPitaevskiiProblem(u₀, lengths::NTuple{N,T}, δt, dispersion!, potential!, nonlinearity, pump!) where {N,T}
        @assert ndims(u₀) == length(lengths) + 1

        sdims = spatial_dims(u₀, lengths)
        ssize = spatial_size(u₀, lengths)

        u = similar(u₀, complex(eltype(u₀)))
        buffer = similar(u)

        rs = (fftfreq(n, l) for (n, l) ∈ zip(ssize, lengths))
        ks = (fftfreq(n, 2π * n / l) for (n, l) ∈ zip(ssize, lengths))

        exp_Aδt = get_exponential(dispersion!, u, ks, δt)
        exp_Vδt = get_exponential(potential!, u, rs, δt / 2)
        G_δt = get_Gδt(nonlinearity, δt / 2)

        plan = plan_fft!(u, sdims)
        iplan = inv(plan)

        T1 = typeof(u)
        T2 = float(typeof(δt))
        T3 = typeof(exp_Aδt)
        T4 = typeof(exp_Vδt)
        T5 = typeof(G_δt)
        T6 = typeof(pump!)
        T7 = typeof(plan)
        T8 = typeof(iplan)

        new{N,T1,T2,T3,T4,T5,T6,T7,T8}(u₀, u, buffer, δt, lengths, exp_Aδt, exp_Vδt, G_δt, pump!, plan, iplan)
    end
end

spatial_dims(prob::GrossPitaevskiiProblem) = spatial_dims(prob.u₀, prob.lengths)
spatial_size(prob::GrossPitaevskiiProblem) = spatial_size(prob.u₀, prob.lengths)

Base.show(io::IO, ::GrossPitaevskiiProblem{N,T1,T2,T3,T4,T5,T6,T7,T8}) where {N,T1,T2,T3,T4,T5,T6,T7,T8} = print(io, "$(N)D GrossPitaevskiiProblem{$T1}")