using Pkg
Pkg.add(["Enzyme", "ForwardDiff", "ReverseDiff", "Zygote"])

using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using SparseConnectivityTracer
using SparseMatrixColorings
using Zygote: Zygote
using Test

## Dense

onearg_backends = [
    SecondOrder(AutoForwardDiff(), AutoZygote()),
    SecondOrder(AutoForwardDiff(), AutoReverseDiff()),
    SecondOrder(AutoReverseDiff(), AutoZygote()),
]

twoarg_backends = [
    SecondOrder(
        AutoForwardDiff(), AutoEnzyme(; mode=Enzyme.Forward, constant_function=true)
    ),
    SecondOrder(
        AutoEnzyme(; mode=Enzyme.Reverse, constant_function=true), AutoForwardDiff()
    ),
]

for backend in vcat(onearg_backends, twoarg_backends)
    @test check_available(backend)
    if backend in onearg_backends
        @test !check_twoarg(backend)
    else
        @test check_twoarg(backend)
    end
    @test check_hessian(backend)
end

test_differentiation(
    vcat(onearg_backends, twoarg_backends); first_order=false, logging=LOGGING
);

## Sparse

sparse_backends = [
    AutoSparse(
        SecondOrder(AutoForwardDiff(), AutoZygote());
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    ),
]

test_differentiation(
    sparse_backends, sparse_scenarios(); first_order=false, sparsity=true, logging=LOGGING
);
