using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest
using DifferentiationInterfaceTest: insert_context
using ForwardDiff: ForwardDiff
using SparseConnectivityTracer
using SparseMatrixColorings
using ComponentArrays: ComponentArrays
using StaticArrays: StaticArrays

LOGGING = get(ENV, "CI", "false") == "false"

## Dense

test_differentiation(AutoForwardDiff(); logging=LOGGING)

## Sparse

sparse_backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)
test_differentiation(sparse_backend, sparse_scenarios(); sparsity=true, logging=LOGGING)

## Contexts

test_differentiation(
    AutoForwardDiff(), insert_context.(default_scenarios()); logging=LOGGING
);
