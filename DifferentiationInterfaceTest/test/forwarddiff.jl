using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using SparseConnectivityTracer
using SparseMatrixColorings
using ComponentArrays: ComponentArrays
using StaticArrays: StaticArrays

## Dense

test_differentiation(AutoForwardDiff(); logging=LOGGING)

## Sparse

sparse_backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)
test_differentiation(sparse_backend, sparse_scenarios(); sparsity=true, logging=LOGGING)

## Weird

test_differentiation(
    AutoForwardDiff(),
    vcat(component_scenarios(), static_scenarios());
    correctness=true,
    logging=LOGGING,
)
