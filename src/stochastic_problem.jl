struct DiagonalNoise{Tfunc,Tnoise}
    func::Tfunc
    function DiagonalNoise(func::Tfunc, ::Type{Tnoise}) where {Tfunc,Tnoise}
        new{Tfunc,Tnoise}(func)
    end
end

function (σ!::DiagonalNoise)(ξ, buffer, u, p, t)
    σ!.func(buffer, u, p, t)
    randn!(ξ)
    ξ .*= buffer
end

struct StochasticGrossPitaevskiiProblem{N,T1,T2,T3,T4,T5,T6,T7,T8}
    base_prob::GrossPitaevskiiProblem{N,T1,T2,T3,T4,T5,T6,T7}
    noise_func::T8

    function StochasticGrossPitaevskiiProblem(dispersion!, potential!, nonlinearity, pump!, u0, lengths, param,
        noise_func::T8) where {T8}

        first_u0 = u0[.., 1]
        base_prob = GrossPitaevskiiProblem(dispersion!, potential!, nonlinearity, pump!, first_u0, lengths, param)
        N, T1, T2, T3, T4, T5, T6, T7 = typeof(base_prob).parameters
        new{N,T1,T2,T3,T4,T5,T6,T7,T8}(base_prob, noise_func)
    end
end

function Base.show(io::IO,
    ::StochasticGrossPitaevskiiProblem{N,T1}) where {N,T1}
    print(io, "$(N)D StochasticGrossPitaevskiiProblem{$T1}")
end

function Base.getproperty(prob::StochasticGrossPitaevskiiProblem, s::Symbol)
    if hasproperty(prob.base_prob, s)
        getproperty(prob.base_prob, s)
    else
        getproperty(prob, s)
    end
end

