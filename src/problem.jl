struct GrossPitaevskiiProblem{N,T1,T2,T3,T4,T5,T6,T7}
    u0::T1
    lengths::NTuple{N,T2}
    rs::NTuple{N,Frequencies{T2}}
    ks::NTuple{N,Frequencies{T2}}
    dispersion::T3
    potential::T4
    nonlinearity::T5
    pump::T6
    spatial_dims::NTuple{N,Int}
    spatial_size::NTuple{N,Int}
    param::T7

    function GrossPitaevskiiProblem(dispersion, potential, nonlinearity, pump, u0, lengths, param=())
        _u0 = complex(u0)

        T1 = typeof(_u0)
        T2 = promote_type(float.(typeof.(lengths))...)
        T3 = typeof(dispersion)
        T4 = typeof(potential)
        T5 = typeof(nonlinearity)
        T6 = typeof(pump)
        T7 = typeof(param)

        N = length(lengths)
        spatial_dims = ntuple(n -> n - N + ndims(u0), N)
        spatial_size = ntuple(n -> size(u0, n - N + ndims(u0)), N)

        rs = ntuple(j -> fftfreq(spatial_size[j], T2(lengths[j])), N)
        ks = ntuple(j -> fftfreq(spatial_size[j], T2(2π * spatial_size[j] / lengths[j])), N)

        new{N,T1,T2,T3,T4,T5,T6,T7}(_u0, lengths, rs, ks,
            dispersion, potential, nonlinearity, pump, spatial_dims, spatial_size, param)
    end
end

function Base.show(io::IO,
    ::GrossPitaevskiiProblem{N,T1}) where {N,T1}
    print(io, "$(N)D GrossPitaevskiiProblem{$T1}")
end

function update_initial_condition!(prob::GrossPitaevskiiProblem, new_u0)
    copy!(prob.u0, new_u0)
end