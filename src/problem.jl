struct GrossPitaevskiiProblem{N,T1,T2,T3,T4,T5,T6,T7}
    u₀::T1
    lengths::NTuple{N,T2}
    rs::NTuple{N,Frequencies{T2}}
    ks::NTuple{N,Frequencies{T2}}
    dispersion!::T3
    potential!::T4
    nonlinearity::T5
    pump!::T6
    spatial_dims::NTuple{N,Int}
    spatial_size::NTuple{N,Int}
    param::T7

    function GrossPitaevskiiProblem(dispersion!, potential!, nonlinearity, pump!, u₀, lengths, param=())
        _u₀ = complex(u₀)

        T1 = typeof(_u₀)
        T2 = promote_type(float.(typeof.(lengths))...)
        T3 = typeof(dispersion!)
        T4 = typeof(potential!)
        T5 = typeof(nonlinearity)
        T6 = typeof(pump!)
        T7 = typeof(param)

        N = length(lengths)
        spatial_dims = ntuple(n -> n - N + ndims(u₀), N)
        spatial_size = ntuple(n -> size(u₀, n - N + ndims(u₀)), N)

        rs = ntuple(j -> fftfreq(spatial_size[j], T2(lengths[j])), N)
        ks = ntuple(j -> fftfreq(spatial_size[j], T2(2π * spatial_size[j] / lengths[j])), N)

        new{N,T1,T2,T3,T4,T5,T6,T7}(_u₀, lengths, rs, ks,
            dispersion!, potential!, nonlinearity, pump!, spatial_dims, spatial_size, param)
    end
end

function Base.show(io::IO,
    ::GrossPitaevskiiProblem{N,T1,T2,T3,T4,T5,T6}) where {N,T1,T2,T3,T4,T5,T6}
    print(io, "$(N)D GrossPitaevskiiProblem{$T1}")
end

function update_initial_condition!(prob::GrossPitaevskiiProblem, new_u₀)
    copy!(prob.u₀, new_u₀)
end