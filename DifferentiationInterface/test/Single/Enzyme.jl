using ADTypes: ADTypes
using DifferentiationInterface, DifferentiationInterfaceTest
using Enzyme: Enzyme
using SparseConnectivityTracer, SparseMatrixColorings
using Test

dense_backends = [
    AutoEnzyme(; mode=nothing),
    AutoEnzyme(; mode=Enzyme.Forward),
    AutoEnzyme(; mode=Enzyme.Reverse),
]

sparse_backends =
    AutoSparse.(
        dense_backends,
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    )

@testset "Check $(typeof(backend))" for backend in vcat(dense_backends, sparse_backends)
    @test check_available(backend)
    @test check_twoarg(backend)
    if ADTypes.mode(backend) isa ADTypes.ForwardOrReverseMode
        @test check_hessian(backend; verbose=false)
    else
        @test_broken check_hessian(backend; verbose=false)
    end
end

## Dense backends

test_differentiation(dense_backends; second_order=false, logging=LOGGING);

test_differentiation(
    [
        AutoEnzyme(; mode=nothing),
        AutoEnzyme(; mode=Enzyme.Reverse),
        SecondOrder(AutoEnzyme(; mode=Enzyme.Reverse), AutoEnzyme(; mode=Enzyme.Reverse)),
        SecondOrder(AutoEnzyme(; mode=Enzyme.Forward), AutoEnzyme(; mode=Enzyme.Reverse)),
    ];
    first_order=false,
    excluded=[SecondDerivativeScenario],
    logging=LOGGING,
);

test_differentiation(
    [AutoEnzyme(; mode=nothing), AutoEnzyme(; mode=Enzyme.Forward)];
    first_order=false,
    excluded=[HessianScenario, HVPScenario],
    logging=LOGGING,
);

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
