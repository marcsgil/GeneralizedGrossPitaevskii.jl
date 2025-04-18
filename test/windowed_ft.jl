"""
This test is based on the fact that, for a field in the vacuum state,
the expectation value of the commutator is zero.

One can then relate the value of the expectation value of the 
average from truncated Wigner with an analytic expression:

⟨ψ^†_1(k′)ψ_2(k)⟩_W =  Σ e^(i(k - k′)r) w1(r) w2(r)^* / 2L

where

ψ_j(k) = Σ ψ(r) w_j(r) e^(-ikr) / N

and w1 and w2 are the window functions, L is the length of the system
and N is the number of points in the discretization.

The fields are normalized such that

[ψ(r), ψ′(r′)] = δ_rr′ N / L
"""

@testset "Windowed FT" begin
    function dispersion(ks, param)
        -im * param.γ / 2 + param.ħ * sum(abs2, ks) / 2param.m - param.δ₀
    end

    function noise_func(ψ, param)
        √(param.γ / 2 / param.δL)
    end

    function correlation(sol, rs, window1, par1, window2, par2)
        sol1 = sol .* map(x -> window1(x, par1), rs)
        sol2 = sol .* map(x -> window2(x, par2), rs)

        ft_sol1 = ifftshift(fft(fftshift(sol1, 1), 1), 1)
        ft_sol2 = ifftshift(fft(fftshift(sol2, 1), 1), 1)

        g1 = similar(sol, size(sol, 1), size(sol, 1))

        Threads.@threads for j in axes(g1, 2)
            j_slice = view(ft_sol2, j, :)
            for i in axes(g1, 1)
                i_slice = view(ft_sol1, i, :)
                g1[i, j] = j_slice ⋅ i_slice / length(sol)
            end
        end

        g1
    end

    function analytic_commutation(L, N, window1, par1, window2, par2)
        rs = range(; start=-L / 2, step=L / N, length=N)
        ks = range(-π / L, step=2π / L, length=N)
        [
            sum(rs) do r
                cis((k - k′) * r) * window1(r, par1) * conj(window2(r, par2))
            end for k in ks, k′ in ks
        ] / (rs[end] - rs[1]) / 2
    end

    window(x, (x0, w)) = exp(-(x - x0)^2 / w^2)

    # Space parameters
    L = 20.0f0
    lengths = (L,)
    N = 64
    δL = L / N
    rs = range(; start=-L / 2, step=L / N, length=N)

    # Polariton parameters
    ħ = 0.6582f0 #meV.ps
    γ = 0.1f0 / ħ
    m = ħ^2 / 2.5f0
    δ₀ = 0.49 / ħ

    dt = 4.0f0

    # Full parameter tuple
    param = (; δ₀, m, γ, ħ, L, δL, N, dt)

    u0 = (zeros(ComplexF32, N, 10^4),)
    noise_prototype = similar.(u0)

    prob = GrossPitaevskiiProblem(u0, lengths; dispersion, param, noise_func, noise_prototype)
    tspan = (0, 200.0f0)
    nsaves = 1
    alg = StrangSplitting()
    ts, _sol = GeneralizedGrossPitaevskii.solve(prob, alg, tspan; nsaves, dt, save_start=false, show_progress=false, rng=Random.MersenneTwister(1234))
    ts, _sol = GeneralizedGrossPitaevskii.solve(prob, alg, tspan; nsaves, dt, save_start=false, show_progress=false)
    sol = dropdims(_sol[1], dims=3)

    for x0 ∈ -1:1, w ∈ 3:5
        args = (window, (x0, w), window, (-x0, w))

        corr = correlation(sol, rs, args...)
        an_corr = analytic_commutation(L, N, args...)

        @test isapprox(corr, an_corr; rtol=7e-2)
    end
end