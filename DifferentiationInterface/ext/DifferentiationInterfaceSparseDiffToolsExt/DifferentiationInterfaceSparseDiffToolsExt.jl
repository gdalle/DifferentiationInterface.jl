module DifferentiationInterfaceSparseDiffToolsExt

using ADTypes
import DifferentiationInterface as DI
using DifferentiationInterface:
    AnyAutoSparse, HessianExtras, JacobianExtras, NoHessianExtras, SecondOrder, inner, outer
using SparseDiffTools:
    JacPrototypeSparsityDetection,
    SymbolicsSparsityDetection,
    sparse_jacobian,
    sparse_jacobian!,
    sparse_jacobian_cache
using Symbolics: Symbolics

AnyAutoSparseNoSymbolic = Union{
    AutoSparseFiniteDiff,
    AutoSparseForwardDiff,
    AutoSparsePolyesterForwardDiff,
    AutoSparseReverseDiff,
    AutoSparseZygote,
}

include("onearg.jl")
include("twoarg.jl")

end
