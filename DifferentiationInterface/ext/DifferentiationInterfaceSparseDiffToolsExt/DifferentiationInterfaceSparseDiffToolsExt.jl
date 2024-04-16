module DifferentiationInterfaceSparseDiffToolsExt

using ADTypes
import DifferentiationInterface as DI
using DifferentiationInterface:
    HessianExtras, JacobianExtras, NoHessianExtras, SecondOrder, inner, outer
using SparseDiffTools:
    JacPrototypeSparsityDetection,
    SymbolicsSparsityDetection,
    sparse_jacobian,
    sparse_jacobian!,
    sparse_jacobian_cache
using Symbolics: Symbolics

AnyOneArgAutoSparse = Union{
    AutoSparseFiniteDiff,
    AutoSparseForwardDiff,
    AutoSparsePolyesterForwardDiff,
    AutoSparseReverseDiff,
    AutoSparseZygote,
}

AnyTwoArgAutoSparse = Union{
    AutoSparseFiniteDiff,
    AutoSparseForwardDiff,
    AutoSparsePolyesterForwardDiff,
    AutoSparseReverseDiff,
}

DI.check_available(::AnyOneArgAutoSparse) = true

dense(::AutoSparseFiniteDiff) = AutoFiniteDiff()
dense(backend::AutoSparseReverseDiff) = AutoReverseDiff(backend.compile)
dense(::AutoSparseZygote) = AutoZygote()

function dense(backend::AutoSparseForwardDiff{chunksize,T}) where {chunksize,T}
    return AutoForwardDiff{chunksize,T}(backend.tag)
end

function dense(::AutoSparsePolyesterForwardDiff{chunksize}) where {chunksize}
    return AutoSparsePolyesterForwardDiff{chunksize}()
end

include("onearg.jl")
include("twoarg.jl")

end
