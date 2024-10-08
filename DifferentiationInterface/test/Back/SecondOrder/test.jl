using Pkg
Pkg.add(["ForwardDiff", "ReverseDiff", "Zygote"])

using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using SparseConnectivityTracer
using SparseMatrixColorings
using Zygote: Zygote
using Test

LOGGING = get(ENV, "CI", "false") == "false"

## Dense

backends = [
    SecondOrder(AutoForwardDiff(), AutoZygote()),
    SecondOrder(AutoForwardDiff(), AutoReverseDiff()),
    SecondOrder(AutoReverseDiff(), AutoZygote()),
]

sparse_backends =
    AutoSparse.(
        backends;
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    )

for backend in backends
    @test check_available(backend)
end

test_differentiation(backends; first_order=false, logging=LOGGING);

test_differentiation(
    sparse_backends, sparse_scenarios(); first_order=false, sparsity=true, logging=LOGGING
);
