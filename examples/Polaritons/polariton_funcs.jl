function eq_of_state(n, δ, g, γ)
    n * (γ^2 / 4 + (g * n - δ)^2)
end

function dispersion_relation(k, kₚ, g, n₀, δ, m, branch::Bool)
    v = ħ * kₚ / m
    pm = branch ? 1 : -1
    gn₀ = g * n₀
    v * k + pm * √(ħ^2 * k^4 / 4m^2 + (ħ * (2 * g * n₀ - δ) / m) * k^2 + (gn₀ - δ) * (3gn₀ - δ))
end

function speed_of_sound(n, δ, g, ħ, m)
    2g * n ≥ δ ? √(ħ * (2g * n - δ) / m) : NaN
end