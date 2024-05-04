using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest
using DifferentiationInterfaceTest: AutoZeroForward, AutoZeroReverse

using Aqua: Aqua
using JET: JET
using JuliaFormatter: JuliaFormatter
using Test

using DataFrames: DataFrames
using SparseArrays: SparseArrays

using SparseConnectivityTracer: SparseConnectivityTracer
using ForwardDiff: ForwardDiff

function MyAutoSparse(backend::AbstractADType)
    coloring_algorithm = DifferentiationInterface.GreedyColoringAlgorithm()
    sparsity_detector = SparseConnectivityTracer.TracerSparsityDetector()
    return AutoSparse(backend; sparsity_detector, coloring_algorithm)
end
