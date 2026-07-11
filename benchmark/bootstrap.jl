import Pkg

Pkg.activate(@__DIR__)
Pkg.develop(Pkg.PackageSpec(path=normpath(joinpath(@__DIR__, ".."))))
Pkg.instantiate()
