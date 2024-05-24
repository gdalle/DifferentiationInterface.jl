using DifferentiationInterface, DifferentiationInterfaceTest
using SparseConnectivityTracer
using SparseMatrixColorings
using Test
using Zygote: Zygote

backends = [
    AutoChainRules(Zygote.ZygoteRuleConfig()),
    AutoZygote(),
    AutoSparse(
        AutoZygote();
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    ),
]

for backend in backends
    @test check_available(backend)
    @test !check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(backends; excluded=[SecondDerivativeScenario], logging=LOGGING);

test_differentiation(backends[3], sparse_scenarios(); logging=LOGGING)

if VERSION >= v"1.10"
    test_differentiation(
        AutoZygote(),
        vcat(component_scenarios(), gpu_scenarios(), static_scenarios());
        second_order=false,
        logging=LOGGING,
    )
end
