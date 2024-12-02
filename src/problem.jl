spatial_dims(x, ::NTuple{N}) where {N} = ntuple(n -> n - N + ndims(x), N)

get_exponential(::Nothing, u, rs, δt) = nothing

function get_exponential(f!, u, rs, δt)
    dest = Array{eltype(u),ndims(u)+1}(undef, size(u, 1), size(u)...)
    grid_map!(dest, f!, rs...)
    result = stack(exp(slice * δt) for slice ∈ eachslice(dest, dims=(3, 4)))
    get_unionall(u)(result)
end

struct GrossPitaevskiiProblem{N,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10}
    lengths::NTuple{N,T1}
    u₀::T2
    u::T3
    δt::T4
    exp_Aδt::T5
    exp_Vδt::T6
    G_δt::T7
    pump!::T8
    plan::T9
    iplan::T10

    function GrossPitaevskiiProblem(u₀, lengths::NTuple{N,T}, δt, dispersion!, potential!, nonlinearity, pump!) where {N,T}
        @assert ndims(u₀) == length(lengths) + 1

        u = complex.(u₀)
        rs = (LinRange(-l / 2, l / 2, n) for (l, n) ∈ zip(lengths, size(u₀)[begin+1:end]))

        expA_δt = get_exponential(dispersion!, u, rs, δt)
        expV_δt = get_exponential(potential!, u, rs, δt)
        G_δt = nonlinearity * δt

        plan = plan_fft!(u, spatial_dims(u₀, lengths))
        iplan = inv(plan)

        args = (u₀, u, δt, expA_δt, expV_δt, G_δt, pump!, plan, iplan)

        new{N,T,typeof.(args)...}(lengths, args...)
    end
end

Base.show(io::IO, ::GrossPitaevskiiProblem{N,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10}) where {N,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10} = print(io, "$(N)D GrossPitaevskiiProblem")