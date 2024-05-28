using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using SparseConnectivityTracer
using SparseMatrixColorings

sparse_backend = AutoSparse(
    backend;
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

## Dense

test_differentiation(AutoForwardDiff(); logging=LOGGING)

test_differentiation(
    AutoForwardDiff(), component_scenarios(); excluded=[HessianScenario], logging=LOGGING
)

## Sparse

test_differentiation(sparse_backend, sparse_scenarios(); sparsity=true, logging=LOGGING)
