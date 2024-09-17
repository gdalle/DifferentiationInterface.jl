module DifferentiationInterfaceSparseArraysExt

using ADTypes: ADTypes
using Compat
using DifferentiationInterface
using DifferentiationInterface:
    DenseSparsityDetector, PushforwardFast, PushforwardSlow, basis, pushforward_performance
import DifferentiationInterface as DI
using SparseArrays: SparseMatrixCSC, nonzeros, nzrange, rowvals, sparse

include("sparsity_detector.jl")

end
