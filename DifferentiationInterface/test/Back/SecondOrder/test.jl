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

LOGGING = get(ENV, "CI", "false") == "false"

## Dense

onearg_backends = [
    SecondOrder(AutoForwardDiff(), AutoZygote()),
    SecondOrder(AutoReverseDiff(), AutoZygote()),
]

twoarg_backends = [
    SecondOrder(AutoForwardDiff(), AutoReverseDiff(; compile=true)),
    SecondOrder(AutoForwardDiff(; tag=:mytag), AutoReverseDiff(; compile=false)),
    SecondOrder(AutoForwardDiff(), AutoEnzyme(; mode=Enzyme.Forward)),
    SecondOrder(
        AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const),
        AutoForwardDiff(),
    ),
]

for backend in vcat(onearg_backends, twoarg_backends)
    @test check_available(backend)
    if backend in onearg_backends
        @test !check_inplace(backend)
    else
        @test check_inplace(backend)
    end
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
