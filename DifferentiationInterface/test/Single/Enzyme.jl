using DifferentiationInterface, DifferentiationInterfaceTest
using Enzyme: Enzyme
using SparseConnectivityTracer
using SparseMatrixColorings
using Test

dense_backends = [
    AutoEnzyme(; mode=nothing),
    AutoEnzyme(; mode=Enzyme.Forward),
    AutoEnzyme(; mode=Enzyme.Reverse),
]

sparse_backends = [
    AutoSparse(
        AutoEnzyme(; mode=Enzyme.Forward);
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    ),
    AutoSparse(
        AutoEnzyme(; mode=Enzyme.Reverse);
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    ),
]

for backend in vcat(dense_backends, sparse_backends)
    @test check_available(backend)
    @test check_twoarg(backend)
    @test !check_hessian(backend; verbose=false)
end

## Dense backends

test_differentiation(dense_backends; second_order=false, logging=LOGGING);

test_differentiation(dense_backends; first_order=false, logging=LOGGING);  # TODO: fails

test_differentiation(
    AutoEnzyme(; mode=Enzyme.Forward);  # TODO: add more
    correctness=false,
    type_stability=true,
    second_order=false,
    logging=LOGGING,
);

## Sparse backends

test_differentiation(
    sparse_backends,
    default_scenarios();
    excluded=[DerivativeScenario, GradientScenario, PullbackScenario, PushforwardScenario],
    second_order=false,
    logging=LOGGING,
);

test_differentiation(
    sparse_backends, sparse_scenarios(); second_order=false, sparsity=true, logging=LOGGING
);
