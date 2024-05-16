using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using SparseConnectivityTracer

function MyAutoSparse(backend::AbstractADType)
    coloring_algorithm = GreedyColoringAlgorithm()
    sparsity_detector = TracerSparsityDetector()
    return AutoSparse(backend; sparsity_detector, coloring_algorithm)
end

test_differentiation(AutoForwardDiff(); logging=LOGGING == "false")

test_differentiation(
    MyAutoSparse(AutoForwardDiff()),
    sparse_scenarios();
    sparsity=true,
    logging=LOGGING == "false",
)

test_differentiation(
    AutoForwardDiff(),
    component_scenarios();
    excluded=[HessianScenario],
    logging=LOGGING == "false",
)
