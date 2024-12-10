"""
    free_propagation_template(u0, lengths, kz)

Create a `GrossPitaevskiiProblem` for free paraxial propagation with the given initial state 
`u0` and lengths `lengths`.

Solves the equation `2i * kz * ∂_tψ + ∇²ψ = 0`.
"""
function free_propagation_template(u0, lengths, kz=1)
    dispersion(ks, param) = sum(abs2, ks) / 2 / first(param)
    GrossPitaevskiiProblem(u0, lengths, dispersion, nothing, nothing, nothing, kz)
end

"""
    kerr_propagation_template(u0, lengths, g, kz=1)

Create a `GrossPitaevskiiProblem` for Kerr propagation with the given initial state `u0`, lengths `lengths`, and nonlinearity `g`.
This is intended to solve the equation `2i * kz * ∂_tψ + ∇²ψ = g|ψ|²ψ`.
The dispersion relation is given by `dispersion(ks, param=1) = sum(abs2, ks) / 2*first(param)`.
"""
function kerr_propagation_template(u0, lengths, g, kz=1)
    dispersion(ks, param) = sum(abs2, ks) / 2 / first(param)
    GrossPitaevskiiProblem(u0, lengths, dispersion, nothing, g / 2kz, nothing, kz)
end

function lower_polariton_template(u0, lengths, g, δ)
    
end