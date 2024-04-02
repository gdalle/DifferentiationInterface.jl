using Pkg

Pkg.develop(
    Pkg.PackageSpec(;
        path=joinpath(dirname(@__DIR__), "lib", "DifferentiationInterfaceTest")
    ),
)

##

using DifferentiationInterface
using DifferentiationInterfaceTest
using DifferentiationInterfaceTest: AutoZeroForward, AutoZeroReverse

using Aqua: Aqua
using Documenter: Documenter
using JET: JET
using JuliaFormatter: JuliaFormatter
using Test

using Chairmarks: Chairmarks
using DataFrames: DataFrames
using SparseArrays: SparseArrays

##

# using Diffractor: Diffractor
# using Enzyme: Enzyme
# using FastDifferentiation: FastDifferentiation
# using FiniteDiff: FiniteDiff
# using FiniteDifferences: FiniteDifferences
# using ForwardDiff: ForwardDiff
# using PolyesterForwardDiff: PolyesterForwardDiff
# using ReverseDiff: ReverseDiff
# using SparseDiffTools: SparseDiffTools
# using Symbolics: Symbolics
using Tapir: Tapir
# using Tracker: Tracker
# using Zygote: Zygote
