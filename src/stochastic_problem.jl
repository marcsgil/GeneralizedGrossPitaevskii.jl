struct DiagonalNoise end

struct StochasticGrossPitaevskiiProblem{form,Tnoise,N,T1,T2,T3,T4,T5,T6,T7}
    base_prob::GrossPitaevskiiProblem{N,T1,T2,T3,T4,T5,T6,T7}

    function StochasticGrossPitaevskiiProblem(dispersion!, potential!, nonlinearity, pump!, u0, lengths,
        ::Type{form}, ::Type{Tnoise}, param=()) where {form,Tnoise}

        base_prob = GrossPitaevskiiProblem(dispersion!, potential!, nonlinearity, pump!, u0, lengths, param)
        N, T1, T2, T3, T4, T5, T6, T7 = typeof(base_prob).parameters
        new{form,Tnoise,N,T1,T2,T3,T4,T5,T6,T7}(base_prob)
    end
end

function Base.show(io::IO,
    ::StochasticGrossPitaevskiiProblem{form,Tnoise,N,T1}) where {form,Tnoise,N,T1}
    print(io, "$(N)D StochasticGrossPitaevskiiProblem{$T1}")
end