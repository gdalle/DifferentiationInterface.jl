using Pkg

Pkg.develop(
    Pkg.PackageSpec(; path=joinpath(@__DIR__, "..", "..", "DifferentiationInterfaceTest"))
)

##

using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest

using Aqua: Aqua
using Documenter: Documenter
using JET: JET
using JuliaFormatter: JuliaFormatter
using Test

using SparseArrays: SparseArrays

##

using Diffractor: Diffractor
using Enzyme: Enzyme
using FastDifferentiation: FastDifferentiation
using FiniteDiff: FiniteDiff
using FiniteDifferences: FiniteDifferences
using ForwardDiff: ForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
using ReverseDiff: ReverseDiff
using Symbolics: Symbolics
using Tapir: Tapir
using Tracker: Tracker
using Zygote: Zygote
