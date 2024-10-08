module DifferentiationInterfaceSparseArraysExt

using ADTypes: ADTypes
using DifferentiationInterface
using DifferentiationInterface:
    DenseSparsityDetector, PushforwardFast, PushforwardSlow, basis, pushforward_performance
import DifferentiationInterface as DI
using SparseArrays: SparseMatrixCSC, nonzeros, nzrange, rowvals, sparse

include("sparsity_detector.jl")

end
