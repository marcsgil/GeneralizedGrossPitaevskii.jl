"""
    GrossPitaevskiiProblem(u0, lengths; dispersion=additiveIdentity, potential=additiveIdentity,
        nonlinearity=additiveIdentity, pump=additiveIdentity, noise_func=additiveIdentity, 
        noise_prototype=additiveIdentity, param=nothing)

Represents a generalized Gross-Pitaevskii equation problem with specified initial conditions and domain.

This `struct` encapsulates all components needed to define and solve a generalized Gross-Pitaevskii 
equation, which describes the dynamics of quantum fluids such as Bose-Einstein condensates, 
exciton-polariton condensates, and other nonlinear wave phenomena.

The equation to be solved is of the form:

`` i ∂u(r, t)/∂t = D(-i∇)u + V(r)u + G(u)u + F(r, t) + η(ψ) ξ ``

# Arguments
- `u0::Tuple`: Initial conditions for the fields. This should be a tuple of arrays, where each array 
  represents a field in the simulation. The fields should have the same shape.
- `lengths::Tuple`: Physical dimensions of the simulation domain
- `dispersion`: Function defining energy dispersion term. This should have the signature `
  dispersion(k, param)`, where `k` is a tuple representing a point in momentum space 
  and `param` are additional parameters.
- `potential`: Spatial potential function. This should have the signature `potential(r, param)`, where `r` 
is a tuple representing a point in direct space and `param` are additional parameters.
- `nonlinearity`: Function defining nonlinear interaction terms. This should have the signature 
  `nonlinearity(ψ, param)`, where `ψ` is a tuple representing the fields at a point in direct space
  and `param` are additional parameters.
- `pump`: Function defining pump/drive terms, may be time-dependent. This should have the signature 
  `pump(r, param, t)`, where `r` is a tuple representing a point in direct space, `param` are additional 
  parameters, and `t` is time.
- `noise_func`: Function defining stochastic terms. This should have the signature 
  `noise_func(ψ, param)`, where `ψ` is a tuple representing the fields at a point in direct space and 
  `param` are additional parameters.
- `noise_prototype`: Prototype for noise terms
- `param`: Additional parameters used by the component functions

# Examples
```julia
# Free propagation example
L = 10.0
N = 128
u0 = ()
dispersion(ks, param) = sum(abs2, ks) / 2
prob = GrossPitaevskiiProblem(u0, (L, L); dispersion)

# Exciton-polariton example
function dispersion(k, param)
    Dcc = param.ħ * sum(abs2, k) / 2param.m - param.δc - im * param.γc
    Dxx = -param.δx - im * param.γx
    Dxc = param.Ωr
    @SMatrix [Dcc Dxc; Dxc Dxx]
end
nonlinearity(ψ, param) = @SVector [0, param.g * abs2(ψ[2])]
function pump(r, param, t)
    SVector(param.A * exp(-sum(abs2, r) / param.w^2), 0f0)
end
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity, pump, param)
```
"""
struct GrossPitaevskiiProblem{N,M,T1,T2,T3,T4,T5,T6,T7,T8,T9}
    u0::NTuple{M,T1}
    lengths::NTuple{N,T2}
    dispersion::T3
    potential::T4
    nonlinearity::T5
    pump::T6
    noise_func::T7
    noise_prototype::T8
    param::T9
    function GrossPitaevskiiProblem(u0::Tuple, lengths::Tuple; dispersion::T3=additiveIdentity, potential::T4=additiveIdentity,
        nonlinearity::T5=additiveIdentity, pump::T6=additiveIdentity, noise_func::T7=additiveIdentity, noise_prototype::T8=additiveIdentity,
        param::T9=nothing) where {T3,T4,T5,T6,T7,T8,T9}
        @assert all(x -> ndims(x) ≥ length(lengths), u0)
        @assert all(x -> size(x) == size(first(u0)), u0)
        _u0 = complex.(u0)
        _lengths = promote(lengths...)
        N = length(_lengths)
        M = length(_u0)
        T1 = eltype(_u0)
        T2 = typeof(first(_lengths))
        new{N,M,T1,T2,T3,T4,T5,T6,T7,T8,T9}(_u0, _lengths, dispersion, potential, nonlinearity,
            pump, noise_func, noise_prototype, param)
    end
end

function Base.show(io::IO, ::GrossPitaevskiiProblem{N}) where {N}
    print(io, "$(N)D GrossPitaevskiiProblem")
end

direct_grid(L, N) = StepRangeLen(zero(L), L / N, N)

function direct_grid(prob::GrossPitaevskiiProblem)
    map(direct_grid, prob.lengths, size(first(prob.u0)))
end

reciprocal_grid(L, N) = fftfreq(N, oftype(L, 2π) * N / L)

function reciprocal_grid(prob::GrossPitaevskiiProblem{N}) where {N}
    map(reciprocal_grid, prob.lengths, size(first(prob.u0)))
end