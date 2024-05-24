using DifferentiationInterface, DifferentiationInterfaceTest
using Enzyme: Enzyme
using SparseConnectivityTracer
using SparseMatrixColorings
using Test

backends = [
    AutoEnzyme(; mode=nothing),
    AutoEnzyme(; mode=Enzyme.Forward),
    AutoEnzyme(; mode=Enzyme.Reverse),
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

for backend in backends
    @test check_available(backend)
    @test check_twoarg(backend)
    @test !check_hessian(backend; verbose=false)
end

test_differentiation(backends; second_order=false, logging=LOGGING);

test_differentiation(
    backends[4:5], sparse_scenarios(); second_order=false, sparsity=true, logging=LOGGING
);

test_differentiation(
    backends[2];  # TODO: add more
    correctness=false,
    type_stability=true,
    second_order=false,
    logging=LOGGING,
);
