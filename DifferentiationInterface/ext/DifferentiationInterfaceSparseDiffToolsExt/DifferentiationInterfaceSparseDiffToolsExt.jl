module DifferentiationInterfaceSparseDiffToolsExt

using ADTypes
import DifferentiationInterface as DI
using DifferentiationInterface:
    HessianExtras, JacobianExtras, NoHessianExtras, SecondOrder, inner, outer
using SparseDiffTools:
    AutoSparseEnzyme,
    JacPrototypeSparsityDetection,
    SymbolicsSparsityDetection,
    sparse_jacobian,
    sparse_jacobian!,
    sparse_jacobian_cache
using Symbolics: Symbolics

# used with @eval to avoid Unions and thus ambiguities
SPARSE_BACKENDS = [
    AutoSparseEnzyme,
    AutoSparseFiniteDiff,
    AutoSparseForwardDiff,
    AutoSparsePolyesterForwardDiff,
    AutoSparseReverseDiff,
    AutoSparseZygote,
]

include("onearg.jl")
include("twoarg.jl")

end
