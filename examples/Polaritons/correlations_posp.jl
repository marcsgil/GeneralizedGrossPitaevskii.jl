using HDF5, CairoMakie, CUDA, KernelAbstractions, GeneralizedGrossPitaevskii, ProgressMeter, FFTW

function one_point_corr!(dest, sol)
    backend = get_backend(dest)

    @kernel function kernel!(dest, sol)
        j = @index(Global)
        x = 0f0
        for k ∈ axes(sol, 2)
            x += real(prod(sol[j, k]))
        end
        dest[j] += x
    end

    kernel!(backend, 64)(dest, sol, ndrange=length(dest))
    KernelAbstractions.synchronize(backend)
end

function two_point_corr!(dest, sol)
    backend = get_backend(dest)

    @kernel function kernel!(dest, sol)
        j, k = @index(Global, NTuple)
        x = 0f0
        for m ∈ axes(sol, 2)
            x += real(prod(sol[j, m]) * prod(sol[k, m]))
        end
        dest[j, k] += x
    end

    kernel!(backend, 64)(dest, sol, ndrange=size(dest))
    KernelAbstractions.synchronize(backend)
end

function batch_correlation(path)
    h5open(path) do file
        N = size(first(file))[1]
        one_point_r = CUDA.zeros(Float32, N)
        two_point_r = CUDA.zeros(Float32, (N, N))

        one_point_q = CUDA.zeros(Float32, N)
        two_point_q = CUDA.zeros(Float32, (N, N))

        normalization = 0
        @showprogress for obj ∈ file
            sol = reinterpret(SVector{2,ComplexF32}, read(obj)) |> cu
            rsol = reinterpret(reshape, ComplexF32, sol)
            ft_rsol = fftshift(fft(ifftshift(rsol, 1), 1), 1)
            ft_sol = reinterpret(SVector{2,ComplexF32}, ft_rsol)

            one_point_corr!(one_point_r, sol)
            two_point_corr!(two_point_r, sol)
            one_point_corr!(one_point_q, ft_sol)
            two_point_corr!(two_point_q, ft_sol)
            normalization += size(sol, 2)
        end

        g2_r = normalization * two_point_r ./ ( one_point_r * one_point_r')
        g2_q = normalization * two_point_q ./ ( one_point_q * one_point_q')

        return g2_r, g2_q
    end
end
##
path = "/home/stagios/Marcos/LEON_Marcos/Users/Marcos/hawking_posp.h5"

g2_r, g2_q = batch_correlation(path)

g2_r
##
N = size(g2)[1]
J = N÷2-180:N÷2+180
with_theme(theme_latexfonts()) do
    fig = Figure(; size=(730, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"x", ylabel=L"x\prime")
    hm = heatmap!(ax, (Array(real(g2_r)[J, J]) .- 1) * 1e4, colorrange=(-5, 5), colormap=:inferno)
    #hm = heatmap!(ax, (Array(real(g2)) / length(g2) .- 1) * 1e-2, colorrange=(-5, 5), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(x, x\prime) -1 \ \ ( \times 10^{-4})")
    #save("dev_env/g2m1.pdf", fig)
    fig
end
##
extrema(g2 / length(g2)) 