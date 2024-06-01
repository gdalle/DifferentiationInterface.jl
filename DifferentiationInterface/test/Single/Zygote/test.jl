using DifferentiationInterface, DifferentiationInterfaceTest
using SparseConnectivityTracer, SparseMatrixColorings
using Test
using Zygote: Zygote

dense_backends = [AutoChainRules(Zygote.ZygoteRuleConfig()), AutoZygote()]

sparse_backends = [
    AutoSparse(
        AutoZygote();
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

test_differentiation(
    AutoChainRules(Zygote.ZygoteRuleConfig());
    excluded=[SecondDerivativeScenario],
    second_order=VERSION >= v"1.10",
    logging=LOGGING,
);

test_differentiation(AutoZygote(); excluded=[SecondDerivativeScenario], logging=LOGGING);

test_differentiation(
    AutoZygote(),
    vcat(component_scenarios(), static_scenarios());
    second_order=false,
    logging=LOGGING,
)

if VERSION >= v"1.10"
    test_differentiation(AutoZygote(), gpu_scenarios(); second_order=false, logging=LOGGING)
end

## Sparse backends

test_differentiation(
    sparse_backends,
    default_scenarios();
    excluded=[
        DerivativeScenario,
        GradientScenario,
        HVPScenario,
        PullbackScenario,
        PushforwardScenario,
        SecondDerivativeScenario,
    ],
    logging=LOGGING,
);

test_differentiation(sparse_backends, sparse_scenarios(); sparsity=true, logging=LOGGING)
