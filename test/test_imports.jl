using Pkg

Pkg.develop(
    Pkg.PackageSpec(;
        path=joinpath(dirname(@__DIR__), "lib", "DifferentiationInterfaceTest")
    ),
)

Pkg.add(; url="https://github.com/withbayes/Tapir.jl")

##

using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest

using Aqua: Aqua
using Documenter: Documenter
using JET: JET
using JuliaFormatter: JuliaFormatter
using Test

using Chairmarks: Chairmarks
using DataFrames: DataFrames
