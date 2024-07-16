using DifferentiationInterface, DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using SparseConnectivityTracer
using SparseMatrixColorings
using Zygote: Zygote
using Test

dense_backends = [SecondOrder(AutoForwardDiff(), AutoZygote())]

sparse_backends = [
    AutoSparse(
        SecondOrder(AutoForwardDiff(), AutoZygote());
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    ),
]

for backend in vcat(dense_backends, sparse_backends)
    @test check_available(backend)
    @test !check_twoarg(backend)
    @test check_hessian(backend)
end

## Dense backends

test_differentiation(dense_backends; first_order=false, logging=LOGGING);

## Sparse backends

test_differentiation(
    sparse_backends, sparse_scenarios(); first_order=false, sparsity=true, logging=LOGGING
);
