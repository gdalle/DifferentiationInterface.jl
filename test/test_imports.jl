using Pkg

Pkg.develop(
    Pkg.PackageSpec(;
        path=joinpath(dirname(@__DIR__), "lib", "DifferentiationInterfaceTest")
    ),
)

##

using DifferentiationInterface
using DifferentiationInterfaceTest

using Aqua: Aqua
using Documenter: Documenter
using JET: JET
using JuliaFormatter: JuliaFormatter
using Test

using Chairmarks: Chairmarks
using DataFrames: DataFrames
using SparseArrays: SparseArrays
