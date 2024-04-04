using Pkg

Pkg.develop(
    Pkg.PackageSpec(; path=joinpath(@__DIR__, "..", "..", "DifferentiationInterface"))
)

using DifferentiationInterface
using DifferentiationInterfaceTest
using DifferentiationInterfaceTest: AutoZeroForward, AutoZeroReverse

using Aqua: Aqua
using JET: JET
using JuliaFormatter: JuliaFormatter
using Test

using Chairmarks: Chairmarks
using DataFrames: DataFrames
using SparseArrays: SparseArrays

##

using ForwardDiff: ForwardDiff
