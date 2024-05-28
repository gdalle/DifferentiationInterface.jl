using DifferentiationInterface, DifferentiationInterfaceTest
using SparseConnectivityTracer
using SparseMatrixColorings
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

test_differentiation(dense_backends; excluded=[SecondDerivativeScenario], logging=LOGGING);

test_differentiation(
    AutoZygote(),
    vcat(component_scenarios(), gpu_scenarios(), static_scenarios());
    second_order=false,
    logging=LOGGING,
)

## Sparse backends

test_differentiation(
    sparse_backends,
    dense_scenarios();
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
