module DifferentiationInterfaceSparseArraysExt

using ADTypes: ADTypes
using DifferentiationInterface
using DifferentiationInterface:
    DenseSparsityDetector, PushforwardFast, basis, pushforward_performance
import DifferentiationInterface as DI
using SparseArrays: sparse

include("sparsity_detector.jl")

end
